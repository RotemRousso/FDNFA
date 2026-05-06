"""
Korean evaluation runner — outside wrapper around the stock test_results.py
function `test_predicts`. Lives outside the model files so FDNFA's published
predict.py / utils.py / next_frame_classifier.py stay untouched.

Pipeline:
  1. prepare_korean_aligned.py must already have produced
     <korean_root>/test_aligned/  (LH39-mapped .phn and .wrd via panphon).
  2. We call test_predicts twice on the aligned dir:
       - phoneme-level: annotation="phn"
       - word-level:    annotation="wrd"
     `language="english"` because the preprocess already emitted LH39 labels;
     no further dutch_preprocess pass is needed at runtime.

Usage:
    python run_korean_eval.py \
        --wav /home/rotem/projects/datasets/korean/test_aligned \
        --ckpt <path_to_best_model.pt> \
        [--no-plots]
"""
import argparse
import os
import sys

import torch

# Cache torch.load results so the per-file model reload inside main_predict
# costs ~5s ONCE instead of 5s × N_files. Patched only here (outside scripts);
# predict.py / utils.py / next_frame_classifier.py stay byte-identical.
_orig_torch_load = torch.load
_torch_load_cache = {}


def _cached_torch_load(*args, **kwargs):
    path = args[0] if args else kwargs.get("f")
    key = (str(path), kwargs.get("map_location") and "ml")  # path + map flag
    if key not in _torch_load_cache:
        _torch_load_cache[key] = _orig_torch_load(*args, **kwargs)
    return _torch_load_cache[key]


torch.load = _cached_torch_load

# Use the stock evaluator function — single source of truth for the metric.
from test_results import test_predicts


def _print_block(level: str, p10, p15, p20, p25, p50, p100):
    print()
    print("=" * 60)
    print(f"Korean {level.upper()}-LEVEL precision (LH39-mapped via panphon)")
    print("=" * 60)
    print(f"  10 ms:  {p10:6.2f} %")
    print(f"  15 ms:  {p15:6.2f} %")
    print(f"  20 ms:  {p20:6.2f} %")
    print(f"  25 ms:  {p25:6.2f} %")
    print(f"  50 ms:  {p50:6.2f} %")
    print(f" 100 ms:  {p100:6.2f} %")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True,
                        help="Path to the LH39-aligned Korean wav directory "
                             "(produced by prepare_korean_aligned.py).")
    parser.add_argument("--ckpt", required=True,
                        help="Path to the model checkpoint (.pt).")
    parser.add_argument("--w-phi", type=float, default=0.5)
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip per-file diagnostic plots for fast batch runs.")
    parser.add_argument("--levels", nargs="+", default=["phoneme", "word"],
                        choices=["phoneme", "word"])
    args = parser.parse_args()

    if not os.path.isdir(args.wav):
        sys.exit(f"wav dir does not exist: {args.wav}")
    if not os.path.isfile(args.ckpt):
        sys.exit(f"ckpt file does not exist: {args.ckpt}")

    results = {}
    for level in args.levels:
        annotation = "phn" if level == "phoneme" else "wrd"
        print(f"\n>>> Running {level}-level evaluation on {args.wav}")
        p10, p15, p20, p25, p50, p100 = test_predicts(
            wav_dir=args.wav,
            ckpt=args.ckpt,
            w_phi=args.w_phi,
            language="english",   # preprocess already produced LH39 labels
            annotation=annotation,
            no_plots=args.no_plots,
        )
        results[level] = (p10, p15, p20, p25, p50, p100)

    print("\n\n##### Korean evaluation summary #####")
    for level, vals in results.items():
        _print_block(level, *vals)


if __name__ == "__main__":
    main()
