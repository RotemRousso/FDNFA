"""
Evaluate FDNFA on the G2P-orthography Korean dataset (Q2c row of the master
table). Uses the *same* truth source and *same* precision-at-tolerance metric
as compare_mfa_korean.py — so MFA and FDNFA-G2P numbers are computed by
identical code, just on different prediction sources.

Pipeline per utterance:
  1. Read G2P-derived .phn at <g2p_dir>/<name>.phn → phone sequence (LH39)
  2. Call main_predict on the wav with this .phn → predicted boundaries (sec)
  3. Read truth from <truth_dir>/<name>.phn → manual-annotation boundaries (sec)
  4. For each truth boundary, find the closest predicted boundary, count hits
     within each of [10, 15, 20, 25, 50, 100, 500] ms.

Reports both "full data" (utterances with no G2P input penalize FDNFA the same
way they penalize MFA) and "speech-only" (denominator restricted to scored
utterances).

Usage:
    python evaluate_fdnfa_g2p.py \
        --g2p-dir   /home/rotem/projects/datasets/korean/test_aligned_g2p \
        --truth-dir /home/rotem/projects/datasets/korean/test \
        --ckpt      .../Dec2_.../23_best_model.pt \
        --no-plots
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Aggressive no-op patches: predict.py builds 4 large figures per call. Even
# with savefig stubbed, plt.figure/imshow/colorbar are not free at scale (8k
# files). Stub everything heavy to keep batch eval fast.
class _NoOpFig:
    def __getattr__(self, _): return lambda *a, **k: None
_noop = lambda *a, **k: None
plt.figure   = lambda *a, **k: _NoOpFig()
plt.imshow   = _noop
plt.colorbar = _noop
plt.plot     = _noop
plt.bar      = _noop
plt.axvline  = _noop
plt.text     = _noop
plt.xlabel   = _noop
plt.ylabel   = _noop
plt.title    = _noop
plt.legend   = _noop
plt.grid     = _noop
plt.yticks   = _noop
plt.xticks   = _noop
plt.tight_layout = _noop
plt.show     = _noop
plt.savefig  = _noop
plt.close    = _noop
plt.ylim     = lambda *a, **k: (0.0, 1.0)

import torch
from tqdm import tqdm

# Cache torch.load like run_korean_eval.py: ckpt is loaded once for all utts.
_orig_torch_load = torch.load
_torch_load_cache = {}


def _cached_torch_load(*args, **kwargs):
    path = args[0] if args else kwargs.get("f")
    key = (str(path), kwargs.get("map_location") and "ml")
    if key not in _torch_load_cache:
        _torch_load_cache[key] = _orig_torch_load(*args, **kwargs)
    return _torch_load_cache[key]


torch.load = _cached_torch_load

# Import after patching torch.load
from predict import main_predict  # noqa: E402

SR = 16000
THRESHOLDS = [0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.5]


def read_truth_boundaries(path: Path):
    """Interior boundaries (seconds), excluding the last row's end-time."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    if not rows:
        return []
    return [end / SR for (_, end, _) in rows[:-1]]


def precision_metrics(truth_bound, pred_bound):
    """Mirrors compare_mfa_korean.precision_metrics — for each truth bound,
    closest predicted within tolerance counts as a hit."""
    if not truth_bound or len(pred_bound) == 0:
        return 0, {t: 0 for t in THRESHOLDS}
    counts = {t: 0 for t in THRESHOLDS}
    total = 0
    for tb in truth_bound:
        diffs = [abs(float(pb) - tb) for pb in pred_bound]
        closest = min(diffs)
        total += 1
        for t in THRESHOLDS:
            if closest <= t:
                counts[t] += 1
    return total, counts


def evaluate(g2p_dir: Path, truth_dir: Path, ckpt: str, level: str,
             w_phi: float, no_plots: bool):
    truth_ext = "phn" if level == "phoneme" else "wrd"
    annotation = "phn" if level == "phoneme" else "wrd"

    # We iterate over EVERY truth file (full dataset) to penalize utts FDNFA
    # cannot run on (no G2P input available).
    truth_files = sorted(truth_dir.glob(f"*.{truth_ext}"))
    g2p_index = {p.stem: p for p in g2p_dir.glob(f"*.{annotation}")}

    n_with_pred = 0
    n_no_pred = 0
    grand_total = 0
    grand_counts = {t: 0 for t in THRESHOLDS}
    subset_total = 0
    subset_counts = {t: 0 for t in THRESHOLDS}

    for truth_path in tqdm(truth_files, desc=f"{level}"):
        name = truth_path.stem
        truth_bound = read_truth_boundaries(truth_path)
        if not truth_bound:
            continue

        if name not in g2p_index:
            grand_total += len(truth_bound)
            n_no_pred += 1
            continue

        wav_path = g2p_dir / f"{name}.wav"
        if not wav_path.exists():
            grand_total += len(truth_bound)
            n_no_pred += 1
            continue

        try:
            # main_predict reads the .phn next to the wav (in g2p_dir),
            # so its phone sequence is the G2P-derived one.
            pred_bound, _, _ = main_predict(
                str(wav_path), ckpt, w_phi,
                language="dutch",          # multilingual code-path (LH39 mapping idempotent)
                annotation=annotation,
                no_plots=no_plots,
            )
        except Exception as e:
            print(f"  skipped {name}: {e}")
            grand_total += len(truth_bound)
            n_no_pred += 1
            continue

        # main_predict already strips the first prediction in test_predicts,
        # but in our manual scoring we treat all returned bounds equally.
        if hasattr(pred_bound, "tolist"):
            pred_bound = pred_bound.tolist()
        if len(pred_bound) == 0:
            grand_total += len(truth_bound)
            n_no_pred += 1
            continue

        total, counts = precision_metrics(truth_bound, pred_bound)
        grand_total += total
        subset_total += total
        for t, c in counts.items():
            grand_counts[t] += c
            subset_counts[t] += c
        n_with_pred += 1

    return (n_with_pred, n_no_pred, grand_total, grand_counts,
            subset_total, subset_counts)


def _print_block(level, n_with, n_no, total_full, counts_full, total_sub, counts_sub):
    print()
    print("=" * 70)
    print(f"FDNFA(G2P-orthography) Korean {level.upper()}-LEVEL precision")
    print(f"  files w/ FDNFA pred: {n_with}    files skipped: {n_no}")
    print(f"  truth bounds — full: {total_full}, subset: {total_sub}")
    print("=" * 70)
    print(f"  {'tolerance':>10} | {'full data %':>12} | {'speech-only %':>14}")
    print(f"  {'-' * 10} | {'-' * 12} | {'-' * 14}")
    for t in THRESHOLDS:
        full = counts_full[t] * 100.0 / total_full if total_full else 0.0
        sub = counts_sub[t] * 100.0 / total_sub if total_sub else 0.0
        print(f"  {int(t * 1000):>7} ms | {full:>11.2f}% | {sub:>13.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2p-dir", type=Path,
                        default=Path("/home/rotem/projects/datasets/korean/test_aligned_g2p"))
    parser.add_argument("--truth-dir", type=Path,
                        default=Path("/home/rotem/projects/datasets/korean/test"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--w-phi", type=float, default=0.5)
    parser.add_argument("--levels", nargs="+", default=["phoneme", "word"],
                        choices=["phoneme", "word"])
    parser.add_argument("--no-plots", action="store_true", default=True)
    args = parser.parse_args()

    if not args.g2p_dir.is_dir():   sys.exit(f"missing {args.g2p_dir}")
    if not args.truth_dir.is_dir(): sys.exit(f"missing {args.truth_dir}")
    if not os.path.isfile(args.ckpt):
        sys.exit(f"missing ckpt {args.ckpt}")

    for level in args.levels:
        result = evaluate(args.g2p_dir, args.truth_dir, args.ckpt, level,
                          args.w_phi, args.no_plots)
        _print_block(level, *result)


if __name__ == "__main__":
    main()
