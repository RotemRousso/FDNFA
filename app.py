"""
FDNFA web demo — interactive forced alignment in the browser.

Two pretrained checkpoints ship with the repo (under pretrained_models/):
  - fdnfa_timit_english.pt   — TIMIT-trained, best for English phoneme alignment
  - fdnfa_buckeye_english.pt — Buckeye-trained (spontaneous English); also the
                               recommended pick for cross-lingual / multilingual
                               zero-shot alignment (Dutch, German, Hebrew, ...).

The app picks one automatically from the `Language` radio (english → TIMIT,
multilingual → Buckeye); a custom .pt upload overrides both.

For HuggingFace Spaces deployment, set Space Secrets:
  HF_MODEL_REPO   — e.g. "YourUsername/FDNFA-weights"
  HF_TOKEN        — only needed for private repos
The app will download `fdnfa_timit_english.pt` and `fdnfa_buckeye_english.pt`
from that repo on first use.
"""
import os
import re
import shutil
import tempfile
import threading

import gradio as gr
import textgrid
import torchaudio

import utils
from predict import main_predict

# ── Checkpoint configuration ──────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "pretrained_models")

CKPT_FILES = {
    "english":      "fdnfa_timit_english.pt",
    "multilingual": "fdnfa_buckeye_english.pt",
}
CKPT_LABELS = {
    "english":      "TIMIT (English) — best for English phoneme alignment",
    "multilingual": "Buckeye (English, spontaneous) — recommended for cross-lingual / multilingual",
}

_ckpt_cache = {}

def _resolve_ckpt(key: str):
    """Resolve checkpoint path: cache → local file → HF Hub. Returns None if all fail."""
    if key in _ckpt_cache and os.path.exists(_ckpt_cache[key]):
        return _ckpt_cache[key]

    filename = CKPT_FILES[key]
    local_path = os.path.join(PRETRAINED_DIR, filename)
    if os.path.exists(local_path):
        _ckpt_cache[key] = local_path
        return local_path

    repo = os.environ.get("HF_MODEL_REPO", "")
    if repo:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                token=os.environ.get("HF_TOKEN"),
            )
            _ckpt_cache[key] = path
            return path
        except Exception as exc:
            print(f"[FDNFA] HF Hub fetch failed for {filename}: {exc}")

    return None

_inference_lock = threading.Lock()

# ── Internal language routing ────────────────────────────────────────────────

def _internal_language(lang: str, mode: str, ann_ext: str) -> str:
    """
    Map UI choices to the internal `language` flag understood by main_predict.

      'english' = no G2P; assumes labels are already TIMIT-39 phonemes.
      'dutch'   = G2P pipeline (panphon-based articulatory mapping). Used for:
                    • any non-English language
                    • word-level alignment (.wrd, words need phoneme decomposition)
                    • plain text input (could be words or arbitrary phonemes)

    NOTE: `ann_ext` here must be the *original* extension supplied by the user —
    not the post-rewrite extension after a .txt → dummy .phn synthesis.
    """
    if lang == "english" and mode == "phoneme" and ann_ext.lower() == "phn":
        return "english"
    return "dutch"

# ── Core handler ──────────────────────────────────────────────────────────────

OUTPUTS_NONE = (None, None, None, None, None, None, None, None)  # 8 None for the non-status outputs

def run_alignment(audio_file, annotation_file, ckpt_upload, mode, lang,
                  pretrained_choice, w_phi, progress=gr.Progress()):
    if not audio_file or not annotation_file:
        return ("Please upload both an audio file and an annotation file.", *OUTPUTS_NONE)

    ckpt_to_use = ckpt_upload if ckpt_upload else _resolve_ckpt(pretrained_choice)
    if not ckpt_to_use or not os.path.exists(ckpt_to_use):
        return (
            f"No checkpoint found. Expected {CKPT_FILES[pretrained_choice]} "
            f"in {PRETRAINED_DIR}, or HF_MODEL_REPO set, or upload a .pt file.",
            *OUTPUTS_NONE,
        )

    progress(0.1, desc="Preparing workspace...")
    workspace = tempfile.mkdtemp(prefix="fdnfa_")
    base = "input"
    wav_path = os.path.join(workspace, f"{base}.wav")

    original_ext = os.path.basename(annotation_file).split(".")[-1].lower()
    ann_ext = original_ext
    ann_path = os.path.join(workspace, f"{base}.{ann_ext}")

    # Resample audio to 16 kHz mono
    try:
        audio, sr = torchaudio.load(audio_file)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        torchaudio.save(wav_path, audio, 16000)
    except Exception as exc:
        return (f"Audio error: {exc}", *OUTPUTS_NONE)

    shutil.copy(annotation_file, ann_path)

    # Capture the original input tokens (whatever the user supplied per line):
    #   .phn → phoneme labels
    #   .wrd → word labels
    #   .txt → space-separated tokens (words or phonemes)
    if original_ext == "txt":
        with open(ann_path) as f:
            orig_tokens = re.sub(r"[^\w\s]", "", f.read().strip()).split()
        if not orig_tokens:
            return ("Text annotation is empty after stripping punctuation.", *OUTPUTS_NONE)
        # Synthesize a uniform-segments dummy .phn so downstream code has timestamps.
        audio_len = audio.shape[1]
        interval = audio_len / len(orig_tokens)
        ann_ext = "phn"
        ann_path = os.path.join(workspace, f"{base}.{ann_ext}")
        with open(ann_path, "w") as f:
            for i, tok in enumerate(orig_tokens):
                f.write(f"{int(i * interval)} {int((i + 1) * interval)} {tok}\n")
    else:
        with open(ann_path) as f:
            orig_tokens = [ln.strip().split()[-1] for ln in f if ln.strip()]

    # Route by ORIGINAL extension (post-rewrite ann_ext is "phn" for txt inputs).
    language = _internal_language(lang, mode, original_ext)

    progress(0.3, desc="Running alignment...")
    try:
        with _inference_lock:
            utils.set_dp_matrix_out_dir(workspace)
            pred_bound, _truth_bound, mapped_ph = main_predict(
                wav=wav_path,
                ckpt=ckpt_to_use,
                w_phi=w_phi,
                language=language,
                annotation=ann_ext,
            )
            utils.set_dp_matrix_out_dir(None)
    except Exception as exc:
        utils.set_dp_matrix_out_dir(None)
        return (f"Inference error: {exc}", *OUTPUTS_NONE)

    progress(0.8, desc="Building outputs...")

    pred_bound_list = [float(t) for t in pred_bound]

    # ── Build two tables ─────────────────────────────────────────────────────
    # 1) LH39 phonemes  — the aligner's direct output, one row per pred_bound.
    # 2) Original tokens — words (.wrd / .txt) or non-LH39 phonemes (multilingual
    #    .phn). Only populated when the G2P path was used (mapped_ph != None);
    #    for english+phoneme+.phn the LH39 phonemes ARE the original, so the
    #    second table is left empty.
    # Every input token gets a row in the table even if its predicted interval
    # is degenerate; degenerate intervals are still kept out of the TextGrid
    # (Praat rejects zero-length).

    if mapped_ph is not None:
        phn_labels = [ph for seq in mapped_ph for ph in seq]
    else:
        phn_labels = orig_tokens

    table_phonemes, phn_intervals = [], []
    t0 = 0.0
    for i, t1 in enumerate(pred_bound_list):
        lbl = phn_labels[i] if i < len(phn_labels) else ""
        table_phonemes.append([round(t0, 3), round(t1, 3), lbl])
        if t1 > t0:
            phn_intervals.append((t0, t1, lbl))
        t0 = t1

    table_original, orig_intervals = [], []
    if mapped_ph is not None:
        counts_per_token = [len(seq) for seq in mapped_ph]
        cumulative = 0
        t0 = 0.0
        for i, count in enumerate(counts_per_token):
            cumulative += count
            if cumulative - 1 >= len(pred_bound_list):
                break
            t1 = pred_bound_list[cumulative - 1]
            lbl = orig_tokens[i] if i < len(orig_tokens) else ""
            table_original.append([round(t0, 3), round(t1, 3), lbl])
            if t1 > t0:
                orig_intervals.append((t0, t1, lbl))
            t0 = t1

    # ── TextGrid: phones tier always, original tier when applicable ──────────
    max_time = phn_intervals[-1][1] if phn_intervals else 0.0
    tg = textgrid.TextGrid(minTime=0, maxTime=max_time)
    tier_phn = textgrid.IntervalTier(name="phones", minTime=0, maxTime=max_time)
    for t0_iv, t1_iv, lbl_iv in phn_intervals:
        tier_phn.add(minTime=t0_iv, maxTime=t1_iv, mark=lbl_iv)
    tg.append(tier_phn)
    if orig_intervals:
        # Tier name reflects what the original layer represents.
        if mode == "word":
            orig_tier_name = "words"
        elif original_ext == "phn":
            orig_tier_name = "phones_original"
        else:
            orig_tier_name = "tokens"
        tier_orig = textgrid.IntervalTier(name=orig_tier_name, minTime=0, maxTime=max_time)
        for t0_iv, t1_iv, lbl_iv in orig_intervals:
            tier_orig.add(minTime=t0_iv, maxTime=t1_iv, mark=lbl_iv)
        tg.append(tier_orig)
    tg_path = os.path.join(workspace, f"{base}.TextGrid")
    tg.write(tg_path)

    # Status note: the .phn-as-word case produces a second "words" table that
    # actually contains phonemes — surface this in the status so it's not
    # mistaken for a bug.
    status_note = ""
    if mode == "word" and original_ext == "phn":
        status_note = (" Note: input was phoneme-level (.phn) but mode=word "
                       "— the 'original' table shows input phonemes since no "
                       "word annotations were provided.")

    def _img(name):
        p = os.path.join(workspace, name)
        return p if os.path.exists(p) else None

    return (
        "Done." + status_note,
        audio_file,
        _img(f"{base}_boundaries.png"),
        _img("dp_matrix_with_path.png"),
        _img(f"{base}_probs.png"),
        tg_path,
        table_phonemes,
        table_original,
    )

# ── UI ────────────────────────────────────────────────────────────────────────

PRETRAINED_RADIO_CHOICES = [
    (CKPT_LABELS["english"],      "english"),
    (CKPT_LABELS["multilingual"], "multilingual"),
]

with gr.Blocks(title="FDNFA Forced Aligner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FDNFA: Fully Differentiable Neural Forced Aligner")
    gr.Markdown(
        "Upload a speech file and a transcript to predict precise phoneme or word boundaries "
        "using Soft Dynamic Programming."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            audio_in  = gr.Audio(label="Audio file (any sample rate)", type="filepath")
            ann_in    = gr.File(label="Annotation (.phn / .wrd / .txt)")
            mode_in   = gr.Radio(["phoneme", "word"], value="phoneme", label="Mode")
            lang_in   = gr.Radio(["english", "multilingual"], value="english", label="Language")
            pretrained_in = gr.Radio(
                choices=PRETRAINED_RADIO_CHOICES,
                value="english",
                label="Pretrained checkpoint (auto-follows Language; override here if you want)",
            )
            ckpt_in   = gr.File(label="Or upload a custom checkpoint (.pt) — overrides pretrained",
                                file_types=[".pt"], type="filepath")
            wphi_in   = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                  label="φ weight (acoustic ↔ linguistic)")
            btn       = gr.Button("Run Alignment", variant="primary")
            status    = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### Outputs")
            with gr.Tabs():
                with gr.Tab("Alignment Data"):
                    audio_out  = gr.Audio(label="Playback", interactive=False)
                    table_phn_out = gr.Dataframe(
                        headers=["Start (s)", "End (s)", "Phoneme (LH39)"],
                        label="LH39 phonemes — aligner output",
                    )
                    table_orig_out = gr.Dataframe(
                        headers=["Start (s)", "End (s)", "Label"],
                        label="Original input layer (words / non-LH39 phonemes) — empty for English phoneme alignment",
                    )
                    tg_out     = gr.File(label="Download TextGrid (carries both tiers when applicable)")
                with gr.Tab("Visualizations"):
                    img_bounds = gr.Image(label="Predicted Boundaries (latent score + peaks)",
                                         show_download_button=True)
                    img_dp     = gr.Image(label="Soft-DP Matrix",
                                         show_download_button=True)
                    img_probs  = gr.Image(label="BiLSTM Phoneme Probabilities",
                                         show_download_button=True)

    # Auto-flip the pretrained-checkpoint radio when the user changes language.
    lang_in.change(fn=lambda v: v, inputs=lang_in, outputs=pretrained_in)

    btn.click(
        fn=run_alignment,
        inputs=[audio_in, ann_in, ckpt_in, mode_in, lang_in, pretrained_in, wphi_in],
        outputs=[status, audio_out, img_bounds, img_dp, img_probs,
                 tg_out, table_phn_out, table_orig_out],
    )

    # Auto-shutdown the local Python server when the user closes the browser
    # tab. Disabled on HuggingFace Spaces (where SPACE_ID is set automatically)
    # because the container is shared across visitors — one tab close should
    # not tear down everyone else's session.
    if not os.environ.get("SPACE_ID"):
        shutdown_btn = gr.Button("Shutdown", visible=False, elem_id="fdnfa-shutdown-btn")
        shutdown_btn.click(fn=lambda: os._exit(0), inputs=[], outputs=[])
        demo.load(None, None, None, js="""
            () => {
                window.addEventListener('beforeunload', () => {
                    const btn = document.getElementById('fdnfa-shutdown-btn');
                    if (btn) btn.click();
                });
            }
        """)

if __name__ == "__main__":
    demo.launch()
