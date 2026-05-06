"""
Build a Korean MFA-ready corpus directory for `mfa align`.

Reuses parsing logic from datasets/korean/convert_to_timit.py to extract the
orthographic Hangul transcript (`utt.ortho.` tier) for each utterance segment
that the existing `korean/test/` (or train/) directory already exposes as a
.wav file. Writes a sibling `.lab` file containing the cleaned Hangul text.

Output directory layout (compatible with MFA's flat-corpus mode):

    <out_dir>/
        <speaker_id>/
            s37f46m1_utt0000.wav -> ../../korean/test/s37f46m1_utt0000.wav
            s37f46m1_utt0000.lab
            ...

Why a per-speaker subdir? MFA infers speaker from the parent directory name,
which improves alignment quality and matches the conventional MFA layout.

Usage:
    python prepare_korean_for_mfa.py \
        --src /home/rotem/projects/datasets/korean \
        --split test \
        --out /home/rotem/projects/datasets/korean/test_mfa
"""
import argparse
import os
import re
import sys
from pathlib import Path

# Reuse the existing TextGrid parser so we don't duplicate logic.
KOREAN_DIR = Path("/home/rotem/projects/datasets/korean")
sys.path.insert(0, str(KOREAN_DIR))
from convert_to_timit import (  # noqa: E402
    read_textgrid,
    parse_tier,
    parse_all_tiers,
    is_silence_phone,
    SKIP_UTT_LABELS,
)

# Tokens to drop from the orthographic transcript before writing the .lab.
NON_SPEECH_RE = re.compile(r"<[^>]+>")


def speaker_from_stem(stem: str) -> str:
    """sN-prefix match: s37f46m1 → s37."""
    m = re.match(r"(s\d+)", stem)
    return m.group(1) if m else stem


def clean_ortho(text: str) -> str:
    """Strip <SIL>, <NOISE>, <LAUGH>, ... tokens and collapse whitespace."""
    text = NON_SPEECH_RE.sub(" ", text)
    text = text.replace(" ", " ")  # non-breaking space
    return re.sub(r"\s+", " ", text).strip()


def utt_ortho_for_window(ortho_ivls, utt_start, utt_end) -> str:
    """Concatenate ortho-tier intervals that fall within [utt_start, utt_end]."""
    parts = []
    for x1, x2, txt in ortho_ivls:
        # Allow tiny float slack at the boundaries (matches convert_to_timit)
        if x1 >= utt_start - 1e-6 and x2 <= utt_end + 1e-6:
            cleaned = clean_ortho(txt)
            if cleaned:
                parts.append(cleaned)
    return " ".join(parts).strip()


def process_recording(tg_path: Path, split_wav_dir: Path, out_dir: Path,
                      stem: str):
    """Mirror convert_to_timit.py utt-segmentation, but emit MFA .lab files.

    Returns (n_lab_written, n_skipped) — skipped means utterance had no
    usable orthographic transcript (e.g., entirely <NOISE>).
    """
    text = read_textgrid(tg_path)
    phone_ivls = parse_tier(text, "phoneme")
    utt_ivls = parse_tier(text, "utt.prono.")
    ortho_tiers = parse_all_tiers(text, "utt.ortho.")
    if not phone_ivls or not utt_ivls or not ortho_tiers:
        return 0, 0
    # If multiple utt.ortho. tiers exist, prefer the first.
    ortho_ivls = ortho_tiers[0]

    speaker = speaker_from_stem(stem)
    spk_dir = out_dir / speaker
    spk_dir.mkdir(parents=True, exist_ok=True)

    n_written, n_skipped = 0, 0
    utt_count = 0
    for utt_start, utt_end, utt_label in utt_ivls:
        if utt_label in SKIP_UTT_LABELS:
            continue
        if utt_end <= utt_start:
            continue
        utt_phones = [(a, b, p) for (a, b, p) in phone_ivls
                      if a >= utt_start - 1e-6 and b <= utt_end + 1e-6]
        if not utt_phones:
            continue

        name = f"{stem}_utt{utt_count:04d}"
        utt_count += 1

        wav_src = split_wav_dir / f"{name}.wav"
        if not wav_src.exists():
            # Either this split doesn't have it, or convert_to_timit excluded
            # it (should rarely happen if the same SKIP/empty-phones rules apply).
            continue

        ortho_text = utt_ortho_for_window(ortho_ivls, utt_start, utt_end)
        if not ortho_text:
            n_skipped += 1
            continue

        wav_link = spk_dir / f"{name}.wav"
        if not wav_link.exists() and not wav_link.is_symlink():
            os.symlink(wav_src.resolve(), wav_link)
        lab_path = spk_dir / f"{name}.lab"
        lab_path.write_text(ortho_text + "\n", encoding="utf-8")
        n_written += 1

    return n_written, n_skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=KOREAN_DIR,
                        help="Korean dataset root containing test/, train/, extracted/labels/")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output corpus directory (will be created)")
    args = parser.parse_args()

    split_wav_dir = args.src / args.split
    labels_dir = args.src / "extracted" / "labels"
    if not split_wav_dir.is_dir():
        sys.exit(f"missing {split_wav_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"missing {labels_dir} — original TextGrids must be extracted")

    args.out.mkdir(parents=True, exist_ok=True)

    # Determine which stems belong to this split by looking at existing wavs.
    stems = sorted({p.name.split("_utt")[0]
                    for p in split_wav_dir.glob("*.wav")})
    print(f"[{args.split}] {len(stems)} unique speaker stems")

    total_written, total_skipped, total_missing_tg = 0, 0, 0
    for stem in stems:
        tg_path = labels_dir / f"{stem}.TextGrid"
        if not tg_path.exists():
            total_missing_tg += 1
            continue
        n_w, n_s = process_recording(tg_path, split_wav_dir, args.out, stem)
        total_written += n_w
        total_skipped += n_s

    print(f"\nDone. {total_written} .lab files written, "
          f"{total_skipped} skipped (empty ortho), "
          f"{total_missing_tg} missing TextGrids → {args.out}")


if __name__ == "__main__":
    main()
