"""
Build a parallel Korean dataset whose .phn / .wrd labels come from running an
off-the-shelf G2P over the *orthographic* Hangul transcripts (the same input
MFA receives), then mapped Korean phones → IPA → LH39 via the existing pipeline.

Why this exists
---------------
The default `prepare_korean_aligned.py` rewrites labels from the `phoneme`
tier — i.e. manual phonetic annotation. That gives FDNFA a phone sequence
closer to ground truth than what MFA gets (MFA derives phones from Hangul
through a dictionary). To make the comparison apples-to-apples on the input
side, this script gives FDNFA the *same* phone source MFA uses:

    Hangul  →  korean_mfa.zip G2P dict  →  Korean IPA-like phones  →  LH39

The output dir mirrors `test_aligned/`'s flat layout. Truth boundaries for the
metric still come from `test/*.phn` (manual phoneme tier) — we never relabel
truth, only the model's input phone sequence.

Synthetic timings
-----------------
The .phn columns 1–2 (start, end sample) are linearly distributed across the
audio. FDNFA's `main_predict` only uses these for printing `num_peaks` and as
the returned `truth_bound`; the model itself reads only the phone sequence.
The downstream evaluator (`evaluate_fdnfa_g2p.py`) discards the synthetic
truth and scores predictions against `test/*.phn` instead.

Usage:
    python prepare_korean_aligned_from_g2p.py \
        --src-test  /home/rotem/projects/datasets/korean/test \
        --lab-dir   /home/rotem/projects/datasets/korean/test_mfa \
        --dict      /tmp/korean_g2p_dict.txt \
        --out       /home/rotem/projects/datasets/korean/test_aligned_g2p
"""
from __future__ import annotations
import argparse
import os
import sys
import wave
from pathlib import Path

import korean_preprocess as kp


def load_g2p_dict(path: Path) -> dict:
    """Return word → list-of-IPA-phones (first pronunciation only)."""
    d = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        word, phones = parts[0], parts[1].split()
        # First pronunciation wins; later duplicates are ignored
        if word not in d:
            d[word] = phones
    return d


def ipa_to_lh39(ipa_list):
    """Map a sequence of IPA phones to LH39 via the existing korean_preprocess
    panphon pipeline. Each IPA segment becomes one LH39 label."""
    out = []
    for ipa in ipa_list:
        # korean_preprocess.find_best_leehon39 (via aligner_pipeline) takes a
        # single IPA segment and returns the closest LH39 phone.
        from dutch_preprocess import find_best_leehon39
        match, _ = find_best_leehon39(ipa)
        out.append(match)
    return out


def lab_to_lh39(lab_text: str, g2p: dict) -> tuple[list[str], list[list[str]]]:
    """Hangul transcript → flat LH39 phone sequence + per-word LH39 first-phone.

    Returns (phones_flat, words_first_phone) where:
      - phones_flat: one LH39 per phoneme (for .phn)
      - words_first_phone: list of [first_lh39] per word (for .wrd)
    Words missing from the G2P dict become ['sil'] (silence) — affects 13/7484
    types in this corpus.
    """
    phones_flat = []
    words_first_phone = []
    for word in lab_text.split():
        ipa = g2p.get(word)
        if not ipa:
            phones_flat.append("sil")
            words_first_phone.append(["sil"])
            continue
        lh39 = ipa_to_lh39(ipa)
        phones_flat.extend(lh39)
        words_first_phone.append([lh39[0] if lh39 else "sil"])
    return phones_flat, words_first_phone


def wav_num_samples(wav_path: Path) -> int:
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes()


def write_phn(phones: list[str], n_samples: int, out_path: Path):
    """Write .phn with linearly spaced synthetic timings.

    Format: `<start_sample> <end_sample> <label>` per row, like TIMIT.
    Synthetic timings are placeholders — the metric uses real truth from
    test/<name>.phn, not these.
    """
    if not phones:
        out_path.write_text("", encoding="utf-8")
        return
    n = len(phones)
    step = n_samples / n
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ph in enumerate(phones):
            s = int(round(i * step))
            e = int(round((i + 1) * step))
            f.write(f"{s} {e} {ph}\n")


def write_wrd(words: list[list[str]], n_samples: int, out_path: Path):
    """Each row = one word, label = LH39 of word's first phoneme. Same row
    format / synthetic timings as .phn."""
    if not words:
        out_path.write_text("", encoding="utf-8")
        return
    n = len(words)
    step = n_samples / n
    with open(out_path, "w", encoding="utf-8") as f:
        for i, w in enumerate(words):
            s = int(round(i * step))
            e = int(round((i + 1) * step))
            f.write(f"{s} {e} {w[0]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-test", type=Path,
                        default=Path("/home/rotem/projects/datasets/korean/test"))
    parser.add_argument("--lab-dir", type=Path,
                        default=Path("/home/rotem/projects/datasets/korean/test_mfa"))
    parser.add_argument("--dict", type=Path,
                        default=Path("/tmp/korean_g2p_dict.txt"))
    parser.add_argument("--out", type=Path,
                        default=Path("/home/rotem/projects/datasets/korean/test_aligned_g2p"))
    args = parser.parse_args()

    if not args.src_test.is_dir(): sys.exit(f"missing {args.src_test}")
    if not args.lab_dir.is_dir():  sys.exit(f"missing {args.lab_dir}")
    if not args.dict.is_file():    sys.exit(f"missing {args.dict}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"loading G2P dict from {args.dict} ...")
    g2p = load_g2p_dict(args.dict)
    print(f"  {len(g2p)} word→pronunciation entries")

    # Build name → .lab path index across all speakers
    lab_index = {p.stem: p for p in args.lab_dir.rglob("*.lab")}
    print(f"  {len(lab_index)} .lab files indexed")

    n_written, n_no_lab, n_no_wav = 0, 0, 0
    for wav in sorted(args.src_test.glob("*.wav")):
        name = wav.stem
        lab = lab_index.get(name)
        if lab is None:
            n_no_lab += 1
            continue

        # Symlink wav (idempotent)
        out_wav = args.out / f"{name}.wav"
        if not out_wav.exists() and not out_wav.is_symlink():
            os.symlink(wav.resolve(), out_wav)

        n_samples = wav_num_samples(wav)
        text = lab.read_text(encoding="utf-8").strip()
        phones, words = lab_to_lh39(text, g2p)

        write_phn(phones, n_samples, args.out / f"{name}.phn")
        write_wrd(words, n_samples, args.out / f"{name}.wrd")
        n_written += 1

    print(f"\nDone. wrote: {n_written}, no .lab: {n_no_lab}, missing wav: {n_no_wav}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
