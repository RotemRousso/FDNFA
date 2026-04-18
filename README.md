# FDNFA — Fully Differentiable Neural Forced Aligner

[![Paper](https://img.shields.io/badge/Paper-IEEE%202026-blue)](https://arxiv.org/) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**FDNFA** is an end-to-end, fully differentiable neural system for **phoneme-level and word-level (and also multilingual) forced alignment** — given a speech waveform and a known phoneme sequence, it predicts precise phoneme boundary timestamps.

> Rotem Rousso, Eyal Cohen, Joseph Keshet  
> *"Fully Differentiable Neural Phoneme and Word Level Forced Alignment via Soft Dynamic Programming"*  
> IEEE Journal, 2026

---

## Highlights

- 🎯 **Phoneme-level precision** — operates at ~10 ms frame resolution, the finest granularity among neural forced aligners.
- 🔁 **Fully differentiable** — a Soft Dynamic Programming decoder enables gradient flow through the entire alignment pipeline.
- 🌍 **Zero-shot multilingual** — trained on English, generalizes to Dutch, German, and Hebrew without any additional training.
- 📝 **Word-level generalization** — word boundaries derived from phoneme predictions at test time, competitive with word-level neural aligners.
- ⚡ **Outperforms MFA** — surpasses the Montreal Forced Aligner on TIMIT and Buckeye at 25/50/100 ms tolerances.

---

## How It Works

The system has three jointly trained components:

```
Raw waveform (16 kHz)
        │
        ▼
 ┌──────────────────┐
 │ CNN Encoder f_θ  │  learns boundary-sensitive representations
 │ (5 strided conv) │  via MNCE contrastive loss
 └───────┬──────────┘
         │ Z  (frame-level latent features)
         ├──────────────────────────────────────┐
         ▼                                      │ Z
 ┌────────────────────┐                         │
 │ BiLSTM Context g_ψ │  frame-wise phoneme     │
 │ (5 layers, 512 dim)│  posterior probs U      │
 └───────┬────────────┘                         │
         │ U                                    │
         ▼                                      ▼
 ┌───────────────────────────────────────────────────┐
 │        Soft-DP Decoder h_W                        │
 │  φ₁(Z): acoustic boundary sharpness              │
 │  φ₂(U,p): phoneme posterior consistency          │
 │  LogSumExp forward + soft-argmax backtrack       │
 └───────────────────────────────────────────────────┘
         │
         ▼
  Predicted boundary timestamps  ŷ = (ŷ₁, …, ŷₙ)
```

---

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)

### Option 1 — Conda (recommended)
```bash
conda create -n FDNFA python=3.8
conda activate FDNFA
pip install -r requirements.txt
```

### Option 2 — pip only
```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `torchaudio` are pinned to 2.4.1. For a different CUDA version, install them separately first following [PyTorch's official instructions](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Data Format

FDNFA expects data in **TIMIT-style** format:

```
<dataset_root>/
├── train/
│   ├── speaker1_utt1.wav
│   ├── speaker1_utt1.phn     ← required
│   └── ...
├── val/                       ← only for Buckeye-style splits
└── test/
    ├── speaker2_utt1.wav
    ├── speaker2_utt1.phn
    └── ...
```

Each `.phn` file contains one phoneme segment per line:
```
<start_sample>  <end_sample>  <phoneme_label>
0     3050  h#
3050  4559  sh
4559  5723  ix
...
```

Audio must be **16 kHz mono WAV**. Phoneme labels should use the TIMIT 61-phoneme set (automatically mapped to Lee-Hon 39 internally).

Supported datasets:
- **TIMIT** — `data: timit` in config (standard train/test split, 10% of train used for validation)
- **Buckeye** — `data: buckeye` in config (explicit `train/`, `val/`, `test/` dirs)

---

## Training

Edit `conf/config.yaml` to set your dataset paths and output directory:

```yaml
# Data
data: timit                                   # or 'buckeye'
timit_path: /path/to/your/timit/
buckeye_path: /path/to/your/buckeye/

# Output run directory (Hydra timestamped)
hydra:
  run:
    dir: /path/to/your/runs/my_run/${now:%Y-%m-%d_%H-%M-%S}-${exp_name}
```

Then run:

```bash
conda activate FDNFA
python main.py
```

Checkpoints are saved every epoch as `{epoch}_best_model.pt` in the Hydra run directory. Training progress is logged to [Weights & Biases](https://wandb.ai) (project `DCAF2.0`).

To disable W&B logging:
```bash
WANDB_MODE=disabled python main.py
```

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 200 | Total training epochs |
| `batch_size` | 8 | Per-GPU batch size |
| `lr` | 3e-4 | Learning rate (AdamW) |
| `devices` | [7] | GPU device index(es) — **update this** |
| `num_classes` | 39 | Phoneme set size (Lee-Hon 39) |
| `z_dim` | 256 | CNN encoder output dimension |
| `z_proj` | 64 | Projection head output dimension |

---

## Inference — Single File

Run alignment on a single `.wav` file using a trained checkpoint:

```bash
conda activate FDNFA
python predict.py \
  --wav   /path/to/audio.wav \
  --ckpt  /path/to/checkpoint/12_best_model.pt \
  --prominence 0.1 \
  --mode phoneme \
  --lang english \
  --annotation phn
```

**Flags:**

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `--wav` | any path | — | Path to input `.wav` file (16 kHz mono) |
| `--ckpt` | any path | — | Path to trained checkpoint (`.pt`) |
| `--prominence` | float | `None` | Peak detection prominence threshold |
| `--mode` | `phoneme`, `word` | `phoneme` | Alignment granularity. `phoneme` = phoneme-level (default). `word` = word-level alignment (zero-shot, no additional training needed). |
| `--lang` | `english`, `multilingual` | `english` | Language setting. `english` = default English phoneme alignment (same as training). `multilingual` = any non-English language (zero-shot cross-lingual). |
| `--annotation` | any string | `phn` | Annotation file extension to look for next to the `.wav` file. Use the extension that matches your data (e.g. `phn`, `wrd`, `word`). |

The `.wav` file must have a paired annotation file in the same directory, providing the phoneme or word sequence to align to.

**Examples:**

```bash
# English phoneme-level (default)
python predict.py --wav audio.wav --ckpt runs/my_run/12_best_model.pt --prominence 0.1

# English word-level (zero-shot, no additional training)
python predict.py --wav audio.wav --ckpt runs/my_run/12_best_model.pt --mode word --annotation wrd

# Any non-English language, phoneme-level (zero-shot cross-lingual)
python predict.py --wav audio.wav --ckpt runs/my_run/12_best_model.pt --lang multilingual --annotation phn

# Any non-English language, word-level
python predict.py --wav audio.wav --ckpt runs/my_run/12_best_model.pt --mode word --lang multilingual --annotation wrd
```

**Output:**
- Predicted boundary times in seconds (printed to stdout)
- Visualizations saved next to the input WAV: `<basename>_probs.png`, `<basename>_logits.png`, `<basename>_boundaries.png`

---

## Evaluation — Batch

Evaluate a checkpoint over a directory of `.wav` files and report precision at multiple time thresholds:

```bash
conda activate FDNFA
python test_results.py \
  --wav  /path/to/test/wavs/ \
  --ckpt /path/to/run/dir/ \
  --prominence 0.1 \
  --mode phoneme \
  --lang english \
  --annotation phn
```

**Flags:**

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `--wav` | directory path | — | Directory containing `.wav` files to evaluate |
| `--ckpt` | path | — | Checkpoint directory (sweeps `{idx}_best_model.pt`) or a single `.pt` file |
| `--prominence` | float | `None` | Peak detection prominence threshold |
| `--mode` | `phoneme`, `word` | `phoneme` | Alignment granularity — same meaning as in `predict.py` |
| `--lang` | `english`, `multilingual` | `english` | Language setting — same meaning as in `predict.py` |
| `--annotation` | any string | `phn` | Annotation file extension — same meaning as in `predict.py` |

Reports precision at 10, 15, 20, 25, 50, and 100 ms tolerances. Precision plot saved next to the checkpoint.

---

## Multilingual / Cross-Lingual Alignment

FDNFA can align speech in **unseen languages** (Dutch, German, Hebrew, and others) without any additional training. Phonemes from the target language are automatically mapped to the Lee-Hon 39 set used during English training via articulatory feature distance ([PanPhon](https://github.com/dmort27/panphon)).

To use Dutch phoneme annotations (IFA corpus format):
```bash
python predict.py \
  --wav  /path/to/dutch_audio.wav \
  --ckpt /path/to/checkpoint.pt \
  --prominence 0.1
```
Then set `language = "dutch"` in `predict.py` (`main_predict(..., language="dutch")`).

For other languages, ensure `.phn` files use any standard phoneme notation — the `dutch_preprocess.aligner_pipeline()` function handles IFA→IPA→LH39 conversion.

---

## Latent Space Visualization

Visualize how the CNN encoder learns phoneme-boundary-sensitive representations:

```bash
conda activate FDNFA
python visualize_latent_representation.py \
  --wav     /path/to/audio.wav \
  --run-dir /path/to/training/run/directory/
```

Saves four heatmap plots to `<run_dir>/latent_representations/`:
- `latent_epoch0_untrained.png` — CNN features before training
- `latent_best_trained.png` — CNN features at the best epoch
- `latent_comparison_epoch0_vs_best.png` — side-by-side comparison
- `latent_difference_best_minus_epoch0.png` — difference map

See [`LATENT_VISUALIZATION_README.md`](LATENT_VISUALIZATION_README.md) for details.

---

## Results

### Phoneme-level alignment (English)

| Model | Dataset | P@10ms | P@25ms | P@50ms | P@100ms |
|-------|---------|--------|--------|--------|---------|
| MFA (HMM-GMM) | TIMIT | 38.6 | 72.3 | 81.1 | 84.6 |
| **FDNFA (ours)** | **TIMIT** | **39.18** | **83.98** | **95.26** | **98.64** |
| MFA (HMM-GMM) | Buckeye | **35.3** | 60.6 | 68.9 | 72.7 |
| **FDNFA (ours)** | **Buckeye** | 29.44 | **68.69** | **88.5** | **96.60** |

### Zero-shot multilingual phoneme alignment

| Language | Model | P@25ms | P@50ms | P@100ms |
|----------|-------|--------|--------|---------|
| Dutch (IFA) | MFA | 21.8 | 33.9 | 51.0 |
| Dutch (IFA) | **FDNFA** | **38.82** | **55.48** | **73.66** |
| German (PHONDAT) | MFA | 45.8 | 66.8 | 79.2 |
| German (PHONDAT) | **FDNFA** | **47.57** | **69.58** | **83.32** |
| Hebrew | FDNFA only | 27.06 | 41.57 | 59.02 |

*Hebrew: MFA unavailable (no dictionary/acoustic model for this language).*

---

## Project Structure

```
FDNFA/
├── main.py                          # Training entry point
├── solver.py                        # Train/val/test loop
├── next_frame_classifier.py         # Model + loss functions
├── dataloader.py                    # Dataset classes
├── utils.py                         # Soft-DP, metrics, phoneme maps
├── predict.py                       # Single-file inference
├── test_results.py                  # Batch evaluation
├── dutch_preprocess.py              # Cross-lingual phoneme mapping
├── visualize_latent_representation.py
├── conf/
│   └── config.yaml                  # All hyperparameters (Hydra)
├── requirements.txt
├── GEMINI.md                        # Full architecture documentation
└── LATENT_VISUALIZATION_README.md
```

---

## Architecture Documentation

For a detailed technical description of every component — including the MNCE loss formulation, Soft-DP algorithm, and data pipeline — see [`GEMINI.md`](GEMINI.md).

---

## Citation

If you use FDNFA in your research, please cite:

```bibtex
@article{rousso2026fdnfa,
  title     = {Fully Differentiable Neural Phoneme and Word Level Forced Alignment via Soft Dynamic Programming},
  author    = {Rousso, Rotem and Cohen, Eyal and Keshet, Joseph},
  journal   = {IEEE Journal},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
