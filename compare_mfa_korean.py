"""
Compare MFA's Korean alignment to ground truth, using the SAME precision-at-
threshold metric as FDNFA's test_results.py:

    for each truth boundary t:
        find closest predicted boundary p
        if |t - p| <= threshold: count it
    precision_X = num_within_X / total_truth_boundaries * 100

Truth boundaries come from the original /home/rotem/projects/datasets/korean/test/
files (.phn for phoneme level, .wrd for word level — sample indices at 16 kHz).

MFA boundaries come from the TextGrids in test_mfa_aligned/<spk>/<name>.TextGrid
(in seconds, from the `phones` and `words` tiers).

Both sides use the standard "interior boundaries" convention: skip the very last
end-time (= end of audio).

Usage:
    python compare_mfa_korean.py \
        --truth-dir /home/rotem/projects/datasets/korean/test \
        --mfa-dir   /home/rotem/projects/datasets/korean/test_mfa_aligned
"""
import argparse
import os
import re
import sys
from pathlib import Path

SR = 16000
THRESHOLDS = [0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.5]


def read_truth_boundaries(path: Path):
    """Returns list of interior boundaries in seconds (excludes the last row)."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    if not rows:
        return []
    # interior boundaries = end-time of every row except the last
    return [end / SR for (_, end, _) in rows[:-1]]


# Minimal short-form TextGrid parser (same dialect MFA emits — UTF-8 short-text).
_INTERVAL_RE = re.compile(
    r"intervals\s*\[\d+\]:\s*"
    r"xmin\s*=\s*([\d.eE+\-]+)\s*"
    r"xmax\s*=\s*([\d.eE+\-]+)\s*"
    r'text\s*=\s*"([^"]*)"',
    re.MULTILINE,
)
_TIER_RE = re.compile(
    r'item\s*\[\d+\]:\s*'
    r'class\s*=\s*"IntervalTier"\s*'
    r'name\s*=\s*"([^"]+)"\s*'
    r'xmin\s*=\s*[\d.eE+\-]+\s*'
    r'xmax\s*=\s*[\d.eE+\-]+\s*'
    r'intervals:\s*size\s*=\s*\d+\s*'
    r'(.*?)(?=\s*item\s*\[|\Z)',
    re.DOTALL,
)


def read_mfa_boundaries(path: Path, tier_name: str):
    """Returns list of interior boundaries (sec) for the named tier in a MFA TextGrid."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")

    intervals = []
    for tname, body in _TIER_RE.findall(text):
        if tname == tier_name:
            intervals = [(float(a), float(b), c) for a, b, c in _INTERVAL_RE.findall(body)]
            break
    if not intervals:
        return []
    return [b for (_, b, _) in intervals[:-1]]


def precision_metrics(truth_bound, pred_bound):
    """Same metric used in FDNFA test_results.py.

    For each truth boundary, find closest pred; count it within each threshold.
    Returns (totals, precisions_dict).
    """
    if not truth_bound or not pred_bound:
        return 0, {t: 0.0 for t in THRESHOLDS}
    counts = {t: 0 for t in THRESHOLDS}
    total = 0
    for tb in truth_bound:
        diffs = [abs(pb - tb) for pb in pred_bound]
        closest = min(diffs)
        total += 1
        for t in THRESHOLDS:
            if closest <= t:
                counts[t] += 1
    return total, counts


def evaluate(truth_dir: Path, mfa_dir: Path, level: str):
    """Iterate over EVERY truth file in the dataset. Files MFA failed to align
    (or produced an empty tier for) still contribute their truth boundaries to
    the denominator with 0 within-threshold hits — so the reported precision
    reflects the FULL dataset, with skipped utterances penalising MFA.

    Returns:
        n_with_pred       — files scored with a non-empty MFA tier
        n_no_pred         — files where MFA was missing or produced empty tier
        grand_total       — total truth boundaries (full dataset)
        grand_counts      — within-threshold hit counts (numerator)
        truth_bounds_only — same numbers re-computed on the MFA-success subset only
                            (so we can also report apples-to-apples to FDNFA-subset).
    """
    truth_ext = "phn" if level == "phoneme" else "wrd"
    tier = "phones" if level == "phoneme" else "words"

    # Build name → MFA TextGrid lookup once.
    mfa_index = {tg.stem: tg for tg in mfa_dir.rglob("*.TextGrid")}

    n_with_pred = 0
    n_no_pred = 0
    grand_total = 0
    grand_counts = {t: 0 for t in THRESHOLDS}
    # Speech-only (subset where MFA produced output) — same computation but
    # we only accumulate when a prediction exists, so ratio is on the subset.
    subset_total = 0
    subset_counts = {t: 0 for t in THRESHOLDS}

    for truth_path in sorted(truth_dir.glob(f"*.{truth_ext}")):
        name = truth_path.stem
        truth_bound = read_truth_boundaries(truth_path)
        if not truth_bound:
            continue

        tg_path = mfa_index.get(name)
        pred_bound = read_mfa_boundaries(tg_path, tier) if tg_path is not None else []

        if pred_bound:
            total, counts = precision_metrics(truth_bound, pred_bound)
            grand_total += total
            subset_total += total
            for t, c in counts.items():
                grand_counts[t] += c
                subset_counts[t] += c
            n_with_pred += 1
        else:
            # MFA failed / didn't try this utterance → 0 hits, but truth still counts
            grand_total += len(truth_bound)
            n_no_pred += 1

    return (n_with_pred, n_no_pred, grand_total, grand_counts,
            subset_total, subset_counts)


def _print_block(level, n_with, n_no, total_full, counts_full, total_sub, counts_sub):
    print()
    print("=" * 70)
    print(f"MFA Korean {level.upper()}-LEVEL precision")
    print(f"  files w/ MFA output: {n_with}    files MFA skipped/failed: {n_no}")
    print(f"  truth boundaries — full dataset: {total_full}, MFA-success subset: {total_sub}")
    print("=" * 70)
    print(f"  {'tolerance':>10} | {'full data %':>12} | {'speech-only %':>14}")
    print(f"  {'-' * 10} | {'-' * 12} | {'-' * 14}")
    for t in THRESHOLDS:
        full = counts_full[t] * 100.0 / total_full if total_full else 0.0
        sub = counts_sub[t] * 100.0 / total_sub if total_sub else 0.0
        print(f"  {int(t * 1000):>7} ms | {full:>11.2f}% | {sub:>13.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-dir", type=Path,
                        default=Path("/home/rotem/projects/datasets/korean/test"))
    parser.add_argument("--mfa-dir", type=Path,
                        default=Path("/home/rotem/projects/datasets/korean/test_mfa_aligned"))
    parser.add_argument("--levels", nargs="+", default=["phoneme", "word"],
                        choices=["phoneme", "word"])
    args = parser.parse_args()

    if not args.truth_dir.is_dir():
        sys.exit(f"missing truth dir: {args.truth_dir}")
    if not args.mfa_dir.is_dir():
        sys.exit(f"missing MFA dir: {args.mfa_dir}")

    for level in args.levels:
        (n_with, n_no, total_full, counts_full,
         total_sub, counts_sub) = evaluate(args.truth_dir, args.mfa_dir, level)
        _print_block(level, n_with, n_no, total_full, counts_full, total_sub, counts_sub)


if __name__ == "__main__":
    main()
