"""
Build a parallel Korean dataset directory whose .phn / .wrd labels are mapped
to LH39-compatible phonemes via korean_preprocess (panphon distance over IPA).

After running this, the stock FDNFA test_results.py works on the parallel
directory exactly as it would on TIMIT or IFA Dutch:

    python test_results.py --wav <korean_root>/test_aligned/ --ckpt <ckpt> \
        --mode phoneme --lang english

The ORIGINAL korean/test/ and korean/train/ directories are not touched —
those preserve the full word identity (compound phoneme strings) needed by
the demo app and any downstream display.

Usage:
    python prepare_korean_aligned.py \
        --src /home/rotem/projects/datasets/korean \
        --splits test train         # default
        --link-wavs                 # symlink instead of copy (default)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import korean_preprocess as kp


def _map_label_first(label: str) -> str:
    """Map a (possibly compound) Korean label to a single LH39 phoneme.

    Used for .wrd rows where each row is one word and we want one model-ready
    phoneme per row. Picks the FIRST phoneme of the word.
    """
    pipeline_out = kp.aligner_pipeline(label)
    return pipeline_out[0]["lh39"] if pipeline_out else "sil"


def _map_label_single(label: str) -> str:
    """Map a single-phoneme Korean label to LH39.

    Used for .phn rows where each row is already one phoneme. If the input is
    accidentally compound, falls back to the first phoneme.
    """
    pipeline_out = kp.aligner_pipeline(label)
    return pipeline_out[0]["lh39"] if pipeline_out else "sil"


def _convert_phn(src: Path, dst: Path):
    """Rewrite a .phn file with LH39-mapped labels (one row per phoneme)."""
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            parts = line.split()
            if len(parts) < 3:
                continue
            start, end, label = parts[0], parts[1], parts[2]
            f_out.write(f"{start} {end} {_map_label_single(label)}\n")


def _convert_wrd(src: Path, dst: Path):
    """Rewrite a .wrd file with LH39 labels.

    Each output row remains one word (so boundaries stay word-level, the
    semantic the published evaluator expects). The label becomes the LH39
    mapping of the word's first phoneme — a valid model input that gives the
    DP a real signal at the word's onset, while keeping format compatibility
    with the standard 3-column .wrd reader in predict.py.
    """
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            parts = line.split()
            if len(parts) < 3:
                continue
            start, end, label = parts[0], parts[1], parts[2]
            f_out.write(f"{start} {end} {_map_label_first(label)}\n")


def _link_or_copy_wav(src: Path, dst: Path, link: bool):
    if dst.exists() or dst.is_symlink():
        return
    if link:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def process_split(src_root: Path, split: str, link_wavs: bool) -> int:
    src_dir = src_root / split
    if not src_dir.is_dir():
        print(f"[skip] {src_dir} does not exist")
        return 0

    dst_dir = src_root / f"{split}_aligned"
    dst_dir.mkdir(exist_ok=True)

    wavs = sorted(src_dir.glob("*.wav"))
    n_phn, n_wrd = 0, 0
    for wav in wavs:
        _link_or_copy_wav(wav, dst_dir / wav.name, link_wavs)
        phn = wav.with_suffix(".phn")
        wrd = wav.with_suffix(".wrd")
        if phn.exists():
            _convert_phn(phn, dst_dir / phn.name)
            n_phn += 1
        if wrd.exists():
            _convert_wrd(wrd, dst_dir / wrd.name)
            n_wrd += 1

    print(f"[{split}] {len(wavs)} wavs, {n_phn} .phn, {n_wrd} .wrd → {dst_dir}")
    return len(wavs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="/home/rotem/projects/datasets/korean",
                        help="Korean dataset root containing test/ and train/")
    parser.add_argument("--splits", nargs="+", default=["test", "train"])
    parser.add_argument("--copy-wavs", action="store_true",
                        help="Copy WAVs instead of symlinking (default: symlink)")
    args = parser.parse_args()

    src_root = Path(args.src)
    if not src_root.is_dir():
        sys.exit(f"src does not exist: {src_root}")

    total = 0
    for split in args.splits:
        total += process_split(src_root, split, link_wavs=not args.copy_wavs)
    print(f"\nDone. {total} wav files prepared.")


if __name__ == "__main__":
    main()
