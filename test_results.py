import argparse
import dill
from argparse import Namespace
import torch
import torchaudio
from predict import main_predict, resolve_internal_language, resolve_default_ckpt
from next_frame_classifier import NextFrameClassifier
import dutch_preprocess
from tqdm import tqdm

from dataloader import spectral_size
import matplotlib.pyplot as plt
import numpy as np
import os


def test_predicts(wav_dir, ckpt, w_phi, language="english", annotation="phn", no_plots=False):
    total_sum = 0
    num_10 = 0
    num_15 = 0
    num_20 = 0
    num_25 = 0
    num_50 = 0
    num_100 = 0
    num_500 = 0

    wavs = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
    for wav_file_name in tqdm(wavs, desc="Processing WAV files"):
        wav_file_path = os.path.join(wav_dir, wav_file_name)
        try:
            pred_bound, truth_bound, _ = main_predict(
                wav_file_path, ckpt, w_phi,
                language=language, annotation=annotation, no_plots=no_plots,
            )
        except Exception as e:
            print(f"skipped {wav_file_name}: {e}")
            continue

        # Drop the first predicted boundary (utterance start) before pairing
        # against ground-truth boundary timestamps.
        pred_bound = pred_bound[1:]

        for p in truth_bound:
            curr_pred_bound = (pred_bound - p).abs()
            pred_p_diff = min(curr_pred_bound)
            if pred_p_diff <= 0.1:
                num_100 += 1
                if pred_p_diff <= 0.05:
                    num_50 += 1
                    if pred_p_diff <= 0.025:
                        num_25 += 1
                        if pred_p_diff <= 0.02:
                            num_20 += 1
                            if pred_p_diff <= 0.015:
                                num_15 += 1
                                if pred_p_diff <= 0.01:
                                    num_10 += 1
            total_sum += 1
            if pred_p_diff <= 0.5:
                num_500 += 1

    print(" =================== TOTAL STATS =================== ")
    print(f"thresh = 10 ms:  {num_10*100/total_sum} % ")
    print(f"thresh = 15 ms:  {num_15*100/total_sum} % ")
    print(f"thresh = 20 ms:  {num_20*100/total_sum} % ")
    print(f"thresh = 25 ms:  {num_25*100/total_sum} % ")
    print(f"thresh = 50 ms:  {num_50*100/total_sum} % ")
    print(f"thresh = 100 ms:  {num_100*100/total_sum} % ")
    print(f"thresh = 500 ms:  {num_500*100/total_sum} % ")

    return (num_10*100/total_sum,  num_15*100/total_sum,
            num_20*100/total_sum,  num_25*100/total_sum,
            num_50*100/total_sum,  num_100*100/total_sum)


def _list_ckpts(ckpt_arg):
    """Yield (label, path) pairs given either a single .pt file or a directory
    containing `{idx}_best_model.pt` files."""
    if os.path.isdir(ckpt_arg):
        files = [f for f in os.listdir(ckpt_arg) if f.endswith("_best_model.pt")]
        # sort by leading integer if possible
        def _key(f):
            try:
                return int(f.split("_")[0])
            except ValueError:
                return f
        for f in sorted(files, key=_key):
            yield f.replace("_best_model.pt", ""), os.path.join(ckpt_arg, f)
    else:
        yield os.path.basename(ckpt_arg).replace(".pt", ""), ckpt_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch evaluation: precision at multiple ms thresholds.")
    parser.add_argument('--wav', required=True,
                        help='Directory containing .wav files (with paired annotation files).')
    parser.add_argument('--ckpt', default=None,
                        help='Single .pt file or directory of {idx}_best_model.pt to sweep. '
                             'If omitted, uses the bundled TIMIT checkpoint for --lang english '
                             'or Buckeye for --lang multilingual.')
    parser.add_argument('--mode', type=str, default='phoneme', choices=['phoneme', 'word'])
    parser.add_argument('--lang', type=str, default='english', choices=['english', 'multilingual'])
    parser.add_argument('--annotation', type=str, default='phn',
                        help='Annotation file extension (phn / wrd / etc). Default: phn')
    parser.add_argument('--w-phi', type=float, default=0.5,
                        help='Acoustic ↔ linguistic feature weight. Default: 0.5')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip per-file diagnostic plots for faster runs.')
    parser.add_argument('--out-plot', default=None,
                        help='If set (and --ckpt is a directory), save a precision-vs-checkpoint '
                             'plot to this path.')
    args = parser.parse_args()

    ckpt_arg = args.ckpt or resolve_default_ckpt(args.lang)
    language = resolve_internal_language(args.lang, args.mode, args.annotation)

    labels, p10s, p15s, p20s, p25s, p50s, p100s = [], [], [], [], [], [], []
    for label, ckpt_path in _list_ckpts(ckpt_arg):
        print(f"Testing: {ckpt_path}")
        p10, p15, p20, p25, p50, p100 = test_predicts(
            args.wav, ckpt_path, args.w_phi,
            language=language, annotation=args.annotation, no_plots=args.no_plots,
        )
        labels.append(label)
        p10s.append(p10);  p15s.append(p15);  p20s.append(p20)
        p25s.append(p25);  p50s.append(p50);  p100s.append(p100)

    # Best-checkpoint summary at each threshold
    def _best(pcts):
        i = max(range(len(pcts)), key=lambda j: pcts[j])
        return labels[i], pcts[i]

    print("\n=========== BEST CHECKPOINTS ===========")
    for thresh, pcts in (("10ms", p10s), ("15ms", p15s), ("20ms", p20s),
                        ("25ms", p25s), ("50ms", p50s), ("100ms", p100s)):
        lbl, val = _best(pcts)
        print(f"  {thresh:>5}: ckpt {lbl} → {val:.2f}%")

    if args.out_plot and len(labels) > 1:
        plt.figure(figsize=(10, 6))
        for series, name in ((p10s, '10ms'), (p15s, '15ms'), (p20s, '20ms'),
                            (p25s, '25ms'), (p50s, '50ms'), (p100s, '100ms')):
            plt.plot(labels, series, label=f'Precision @ {name}', marker='o')
        plt.xlabel('Checkpoint')
        plt.ylabel('Precision (%)')
        plt.title('Precision at Different Time Thresholds Across Checkpoints')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(args.out_plot)
        print(f"Saved plot to {args.out_plot}")
