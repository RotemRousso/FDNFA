# ARCHITECTURE.md — FDNFA Architecture Document

> **FDNFA** — *Fully Differentiable Neural Forced Aligner* — is a research system for **phoneme-level forced alignment** of speech. Given a speech waveform and a known phoneme sequence, it predicts precise phoneme boundary timestamps. The system is the subject of the paper:
>
> **"Fully Differentiable Neural Phoneme and Word Level Forced Alignment via Soft Dynamic Programming"**
> Rotem Rousso, Eyal Cohen, Joseph Keshet — *IEEE Journal, 2026*

This codebase (originally named DCAF 2.0) is the cleaned final implementation used to produce the paper's results.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [System Overview](#2-system-overview)
3. [Repository Layout](#3-repository-layout)
4. [Data Pipeline](#4-data-pipeline)
5. [Component 1 — Representation Encoder `f_θ`](#5-component-1--representation-encoder-f_θ)
6. [Component 2 — Context Encoder `g_ψ`](#6-component-2--context-encoder-g_ψ)
7. [Component 3 — Soft-DP Decoder `h_W`](#7-component-3--soft-dp-decoder-h_w)
8. [Loss Functions](#8-loss-functions)
9. [Training Loop — `Solver`](#9-training-loop--solver)
10. [Peak Detection & Boundary Decoding](#10-peak-detection--boundary-decoding)
11. [Inference & Evaluation](#11-inference--evaluation)
12. [Dutch / Multilingual Support](#12-dutch--multilingual-support)
13. [Configuration System](#13-configuration-system)
14. [Experiment Tracking](#14-experiment-tracking)
15. [Ablations](#15-ablations)
16. [Key Results](#16-key-results)
17. [Data Formats](#17-data-formats)
18. [Key Hyperparameters](#18-key-hyperparameters)

---

## 1. Problem Formulation

Let **x** ∈ ℝᵀ be a speech waveform of T samples. Let **p** = (p₁, …, pₙ) be the known phoneme sequence of length N. Each pᵢ ∈ 𝒫, the phoneme set (|𝒫| = 39, Lee-Hon mapping).

The **ground-truth alignment** is **y*** = (y₁, …, yₙ), where yᵢ ∈ [1, T] is the start frame of phoneme pᵢ.

**Goal:** given (**x**, **p**), estimate **ŷ** = (ŷ₁, …, ŷₙ) — the phoneme start times.

A **frame-to-phoneme mapping** π : {1,…,Tₛ} → {1,…,N} assigns each latent frame index τ to the phoneme index i = π(τ), where π(τ) = i whenever yᵢ ≤ τ < yᵢ₊₁.

---

## 2. System Overview

The system has **three trainable components** operating in a pipeline:

```
x (raw waveform, 16 kHz)
│
▼
┌─────────────────────────────────────────────────┐
│  Representation Encoder  f_θ                    │
│  5-layer strided CNN → projection → L2-norm     │
│  Output: Z ∈ ℝ^(D × Tₛ)  (frame representations)│
└───────────────────┬─────────────────────────────┘
                    │ Z
                    ├──────────────────────────────────┐
                    ▼                                  │
┌─────────────────────────────────────────────────┐   │ Z
│  Context Encoder  g_ψ                           │   │
│  5-layer BiLSTM → FC(1024 → 39)                 │   │
│  Output: U ∈ ℝ^(|𝒫| × Tₛ)  (phoneme posteriors)│   │
└───────────────────┬─────────────────────────────┘   │
                    │ U                                │
                    ▼                                  ▼
┌──────────────────────────────────────────────────────────┐
│  Soft-DP Decoder  h_W                                    │
│  Jointly uses Z (via φ₁: boundary sharpness)            │
│  and U (via φ₂: phoneme posterior averages)             │
│  Soft forward pass (LogSumExp) + soft backtrack         │
│  Output: ŷ = (ŷ₁,…,ŷₙ)  — boundary times (frames)     │
└──────────────────────────────────────────────────────────┘

Joint training objective:
  L_total = L_MNCE + η·L_CE + μ·L_SoftDP
```

**Why three components?**
- The **CNN encoder** learns local acoustic features sensitive to phoneme transitions (trained via the MNCE contrastive loss).
- The **BiLSTM context encoder** captures long-range temporal dependencies for frame-wise phoneme posterior distributions (trained via cross-entropy).
- The **Soft-DP decoder** makes alignment decisions in a fully differentiable way, so gradients flow back into both encoders through the alignment process — unlike classical Viterbi which is non-differentiable.

---

## 3. Repository Layout

```
FDNFA/
├── main.py                            # Entry point: train + test (Hydra)
├── solver.py                          # Orchestrates train/val/test epochs
├── next_frame_classifier.py           # NextFrameClassifier: f_θ, g_ψ, h_W + losses
├── dataloader.py                      # Dataset classes + collate_fn
├── utils.py                           # Soft-DP, metrics, peak detection, phoneme maps
├── predict.py                         # Single-file inference (CLI)
├── test_results.py                    # Batch evaluation — precision@threshold
├── dutch_preprocess.py                # IFA → IPA → LH39 cross-lingual phoneme mapping
├── visualize_latent_representation.py # CNN feature heatmap: epoch-0 vs. best
├── build_ifa_dutch_timit_words.py     # Dataset builder for IFA Dutch corpus
├── IFA_corpus_reorder.py              # Corpus reordering utility
├── find_keyerrors.py                  # Debug utility
├── find_missing_phonemes_IFADutch.py  # Dutch phoneme inventory checker
├── conf/
│   └── config.yaml                    # Hydra configuration (all hyperparameters)
├── runs/                              # Per-run output dirs (checkpoints, logs)
├── debug_plots/                       # Visualizations for debugging
└── wandb/                             # Weights & Biases artifacts

Variant / ablation files:
├── next_frame_classifier_NCEclassic.py   # Classic InfoNCE (ablation baseline)
├── next_frame_classifier_orig.py         # Earlier architecture version
└── utils bckupJune.py                    # June backup of utils
```

---

## 4. Data Pipeline

**File:** `dataloader.py`

### `WavPhnDataset` (base class)
- Scans a directory for `.wav` files, loads each paired `.phn` annotation.
- `.phn` format (TIMIT-style): `<start_sample> <end_sample> <phoneme_label>` per line.
- Converts sample-level boundaries to **frame-level** using `spectral_size()` and the `len_ratio`.
- Returns `(audio_tensor, segment_frames, phonemes_list, spectral_length, filename)`.

### `spectral_size(wav_len)`
Computes Tₛ — the CNN output length — by simulating the encoder's strided convolution stack:

| Layer | Kernel | Stride |
|-------|--------|--------|
| Conv1 | 10     | 5      |
| Conv2 | 8      | 4      |
| Conv3 | 4      | 2      |
| Conv4 | 4      | 2      |
| Conv5 | 4      | 2      |

At 16 kHz, **len_ratio ≈ 161.34 samples/frame**, i.e., each latent frame spans ≈ 10 ms.

### Dataset splits

| Class | Dataset | Split strategy |
|-------|---------|---------------|
| `TrainTestDataset` | TIMIT | random_split of `train/` at 90/10; `test/` for testing |
| `TrainValTestDataset` | Buckeye | explicit `train/`, `val/`, `test/` dirs; optional `percent` subset |

### `collate_fn_padd`
Pads variable-length audio tensors along the time axis (`pad_sequence`). Segments and phoneme lists are kept as ragged Python lists.

---

## 5. Component 1 — Representation Encoder `f_θ`

**Implemented in:** `NextFrameClassifier.enc` (`next_frame_classifier.py`)

```
Input: x  [B, T_samples]  →  unsqueeze(1)  →  [B, 1, T_samples]
  │
  ▼
CNN (self.enc):
  Conv1d(1, LS, kernel=10, stride=5)   → BN → LeakyReLU
  Conv1d(LS, LS, kernel=8,  stride=4)  → BN → LeakyReLU
  Conv1d(LS, LS, kernel=4,  stride=2)  → BN → LeakyReLU
  Conv1d(LS, LS, kernel=4,  stride=2)  → BN → LeakyReLU
  Conv1d(LS, Z_DIM, kernel=4, stride=2)
  Transpose  →  [B, Tₛ, Z_DIM]
  │
  ▼
Optional Projection Head (z_proj=64 dims):
  Linear mode:  Dropout1d → Linear(Z_DIM, z_proj)
  MLP mode:     Dropout1d → Linear → LeakyReLU → Dropout1d → Linear
  │
  ▼
L2-normalize each frame vector z_τ
Output: Z ∈ ℝ^(Tₛ × D)
```

Architecture matches the encoder from [Kreuk et al., 2020 (UnSupSeg)] and [Schneider et al. (wav2vec 1.0)].

- `LS = latent_dim` if non-zero, else `Z_DIM` (default 256)
- `Z_DIM = 256`, `z_proj = 64`

### MNCE Training Objective

The encoder is trained with the **Modified Noise Contrastive Estimation (MNCE)** loss, designed specifically to separate intra-phoneme frames from boundary frames:

For phoneme pᵢ with duration lᵢ = yᵢ₊₁ − yᵢ:
- **Positive set** 𝒵⁺ᵢ: frames in central 50% of the phoneme (`[yᵢ + 0.25lᵢ, yᵢ + 0.75lᵢ]`) — steady-state region.
- **Negative set** 𝒵⁻ᵢ: frames within δ=±1 frames of the phoneme boundary yᵢ — transition region.

Loss for anchor frame z_τ:

```
L̃_MNCE(z_τ, 𝒵⁻, 𝒵⁺) = -log [
    (Σ_{z⁺∈𝒦_τ} exp(s(z_τ, z⁺)))^α
    ──────────────────────────────────
    (Σ_{z⁻∈𝒩_τ} exp(s(z_τ, z⁻)))^(1-α)
]
```

where:
- `s(·,·)` = cosine similarity × `cosine_coef`
- `α` = learnable parameter (implemented as `w_pos_neg`, learned via softmax)
- N=5 positive and N=5 negative samples per anchor (`num_repeats=5`)

Total objective sums over all frames in all training examples.

**Key difference from InfoNCE (CPC):** classic InfoNCE contrasts the positive against negatives from *other utterances* (random negatives). MNCE uses **hard negatives** sampled specifically from the phoneme transition region, making the loss directly sensitive to boundary quality. Ablations show this is a decisive contributor to performance (see §16).

---

## 6. Component 2 — Context Encoder `g_ψ`

**Implemented in:** `NextFrameClassifier.bi_lstm` + `NextFrameClassifier.fc`

```
Input: Z  [B, Tₛ, z_proj]
  │
  ▼
Bi-LSTM  (5 layers, hidden=512, bidirectional)
  → [B, Tₛ, 1024]
  │
  ▼
FC: Linear(1024 → num_classes=39)
  → logits [B, Tₛ, 39]
  │
  ▼
Softmax  →  U  [B, Tₛ, 39]  (phoneme posterior per frame)
```

- 5 layers chosen as the **minimum depth to avoid mode collapse** on validation.
- Xavier weight initialization for all LSTM weights.

### CE Training Objective

Frame-wise cross-entropy loss using ground-truth phoneme labels derived from `.phn` annotations and the alignment y*:

```
L_CE(z, p) = -Σ_{τ=1}^{Tₛ} log P_ψ(p_{π(τ)} | z_τ)
```

where `π(τ)` maps frame τ to the phoneme index using the ground-truth segmentation.

In code: `frame_labels[i, start:end, label_index] = 1.0` then `F.cross_entropy(logits, frame_labels.argmax(dim=-1))`.

---

## 7. Component 3 — Soft-DP Decoder `h_W`

**Implemented in:** `utils.py` — `phoneme_alignment()`, `compute_phi_1()`, `compute_phi_2()`

The decoder finds the alignment:

```
ŷ* = argmax_{ỹ} h_W(U, Z, p, ỹ)
   = argmax Σᵢ Σ_{tₑ} Σ_{tₛ<tₑ}  W₁·φ₁(Z, ỹᵢ) + W₂·φ₂(U, pᵢ, ỹᵢ, ỹᵢ₊₁)
```

Weights `W₁`, `W₂` correspond to `w_phi` (learnable `nn.Parameter([0.5, 0.5])`), optimized through the end-to-end gradient.

### Feature Function φ₁ — Acoustic Boundary Sharpness

Captures how much the cosine similarity score **changes** at candidate boundary frame τ:

```
φ₁(Z, ỹᵢ) = d/dτ [ s(z_τ, z_{τ+1}) ] at τ=ỹᵢ
```

In practice, uses a differentiable zero-crossing sharpness measure (tanh-scaled):

```python
# compute_phi_1 in utils.py
score = (1 - sqrt(center²)) + sqrt(Δprev² + ε) + sqrt(Δnext² + ε)
```
applied at both the start frame and end frame of the candidate segment. High scores mark locations where the contrastive similarity curve has a sharp inflection — i.e., phoneme transition points.

### Feature Function φ₂ — Phoneme Posterior Consistency

Average BiLSTM probability of the correct phoneme pᵢ over the candidate segment [tₛ, tₑ]:

```
φ₂(U, pᵢ, tₛ, tₑ) = (1 / (tₑ - tₛ)) · Σ_{j=tₛ}^{tₑ} g_{pᵢ}(z_j)
```

Implemented via cumulative sum of U for efficiency (`compute_phi_2`). Normalizing by segment length prevents the decoder from simply preferring longer segments.

### Soft-DP Forward Pass

DP table V ∈ ℝ^(N × Tₛ × Tₛ): `V[i, tₑ, tₛ]` = best accumulated score for phoneme i ending at frame tₑ, starting at tₛ.

**Soft recursion** (replaces hard max with LogSumExp, temperature γ=1e-20):

```
V_{i,tₑ,tₛ} = D_{i,tₑ,tₛ} + γ · log Σ_{t_prev=0}^{tₛ-1} exp(V_{i-1, tₛ, t_prev} / γ)
```

where `D_{i,tₑ,tₛ} = W₁·φ₁(t_s) + W₂·φ₂(pᵢ, tₛ, tₑ)`.

γ → 0 recovers hard argmax; γ > 0 maintains gradient flow. Value γ=1e-20 chosen empirically from {1e-1, 1e-3, 1e-4, 1e-20}.

### Soft Backtracking (Differentiable)

Standard backtracking uses argmax — non-differentiable. Instead, the predicted boundary ŷᵢ is computed as an **expected value** under the softmax distribution over start frames:

```
α_{i,tₑ,tₛ} = softmax(V_{i,tₑ, 0:tₑ-1} / γ)
ŷᵢ = Σ_{tₛ=0}^{tₑ-1} α_{i,tₑ,tₛ · tₛ
```

This expected-value backtrack allows the MSE loss on boundary timestamps to propagate gradients through the entire pipeline.

---

## 8. Loss Functions

All loss functions are methods on `NextFrameClassifier` in `next_frame_classifier.py`.

### Overall Objective

```
L_total = L_MNCE + η·L_CE + μ·L_SoftDP
```

η = 2×10⁻⁹, μ = 10⁻⁴ (set on TIMIT validation set).

### `loss_ph` (warm-up, epochs 0–4)
Used during initial training before the Soft-DP module is reliable:
- **ph_loss**: cross-entropy on frame-wise phoneme logits.
- **NCE term**: weighted log-softmax over positive/negative cosine scores.
- `total_loss = NCE_loss + 0.2 × ph_loss`
- When segments are provided, boundary MSE is computed but total_loss is reset to `ph_loss` only (pure supervised warm-up).

### `total_loss` (main, epochs 5+)
```
total_loss = NCE_loss + 0.2 × ph_loss + boundary_MSE
```
- **NCE_loss**: for each pred_step t, log-softmax over [pos_score, neg_score] pairs, weighted by `softmax(w_pos_neg)`.
- **boundary_MSE**: `MSE(peak_times_sec, ground_truth_times_sec)`, where segments are converted to seconds via `len_ratio/sr`.

### Ablation loss variants
| Method | Formula | Notes |
|---|---|---|
| `loss_nce` | NCE only | No ph_loss |
| `loss_mse` | NCE + 0.2×ph_loss, replace with MSE | MSE replaces total when peaks available |
| `loss_InfoNCE_classic` | CrossEntropy([pos,neg], target=0) | Classic CPC-style InfoNCE |

---

## 9. Training Loop — `Solver`

**File:** `solver.py`

`Solver(torch.nn.Module)` manages the full lifecycle.

### Initialization
1. `build_model()` — instantiates `NextFrameClassifier`.
2. `prepare_data()` — creates DataLoaders for train/val/test depending on `cfg.data`.
3. `configure_optimizers()` — supports `sgd`, `adam`, `adamW`, `ranger`. Default: `adamW` at lr=3e-4. Schedule: `StepLR(step_size=1000, gamma=0.95)`.

### Loss Curriculum
| Epochs | Loss used | Rationale |
|--------|-----------|-----------|
| 0–4    | `loss_ph` | Warm-up: supervise phoneme classifier before DP is useful |
| 5+     | `total_loss` | Full joint objective: NCE + ph_loss + boundary MSE |

### Epoch Flow
```python
for batch in loader:
    result = forward(batch, mode)     # computes loss + updates stats
    if train:
        result['loss'].backward()
        optimizer.step()
scheduler.step()
generic_eval_end(mode, epoch)         # logs all metrics to W&B
```

### Checkpointing
Saves `{epoch}_best_model.pt` every epoch:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'hparams': argparse.Namespace,
    'peak_detection_params': dill.dumps(defaultdict(...))
}
```
Keys `NFC.enc.*`, `NFC.bi_lstm.*`, `NFC.fc.*`, `NFC.w_phi`, `NFC.w_pos_neg`.

---

## 10. Peak Detection & Boundary Decoding

**File:** `utils.py` — `detect_peaks()` → `detect_peaks_worker()` → `phoneme_alignment()`

### Pipeline
```
Contrastive scores z  [Tₛ]
  → max_min_norm(z)
  → z - median(z)          # center around zero
  → phoneme_alignment()    # Soft-DP (described in §7)
  → [peak frame indices]
  → × len_ratio / sr       # convert to seconds (÷ 16000 × 161.34)
```

### `detect_peaks(x, w_phi, original_lengths, phonemes, len_ratio, probs_real)`
Thin wrapper using `ThreadPoolExecutor(max_workers=1)` for potential future parallelism. Calls `detect_peaks_worker` which normalizes the signal and invokes `phoneme_alignment`.

### `PrecisionRecallMetric`
Tracks raw `(seg, pos_pred, length)` tuples, then at `get_stats()` time runs `detect_peaks` and computes L1/L2 distance between predicted and ground-truth boundaries. Used internally during training for metric logging.

---

## 11. Inference & Evaluation

### `predict.py` — Single-file inference
```bash
python predict.py --wav <path.wav> --ckpt <path.pt> --prominence 0.1
```
Steps:
1. Load checkpoint → reconstruct `NextFrameClassifier` from `ckpt['hparams']`.
2. Load paired `.phn` file; convert phoneme labels (including Dutch cross-lingual mapping if `language="dutch"`).
3. Run forward pass with `torch.no_grad()`.
4. `preds_peaks` from model → `detect_peaks` → predicted boundary times in seconds.
5. Plots: probability heatmap U, predicted vs. truth boundaries.
6. Returns `(pred_boundaries_sec, truth_boundaries_sec)`.

### `test_results.py` — Batch evaluation
```bash
python test_results.py --ckpt <dir_or_pt> --prominence 0.1
```
Iterates over all `.wav` files in `wav_dir`, calls `main_predict`, accumulates counts, and reports:

| Threshold | Metric |
|---|---|
| 10 ms  | P@10  |
| 15 ms  | P@15  |
| 20 ms  | P@20  |
| 25 ms  | P@25  |
| 50 ms  | P@50  |
| 100 ms | P@100 |

For each ground-truth boundary, the closest predicted boundary is found; counts as a hit if within the threshold. Can sweep across checkpoint indices to find the best epoch.

### `visualize_latent_representation.py`
Loads epoch-0 and best-epoch checkpoints, extracts CNN encoder features (Z) for a given `.wav`, saves heatmap comparisons showing how latent representations evolve during training. Outputs saved to `<run_dir>/latent_representations/`.

---

## 12. Dutch / Multilingual Support

**File:** `dutch_preprocess.py`

For multilingual generalization, unseen-language phonemes are mapped to the Lee-Hon 39 set used during training via phonetic feature distances.

### Pipeline (`aligner_pipeline`)
```
IFA label
  → get_ipa_from_ifa()    # IFA notation → IPA symbol(s)
  → find_best_leehon39()  # IPA → closest LH39 via panphon feature_edit_distance
  → LH39 label
```

- `IFA_TO_IPA`: hand-crafted IFA→IPA table (Dutch-specific phoneme notation).
- `LH39_IPA`: maps each Lee-Hon 39 phoneme to its IPA string.
- `find_best_leehon39`: uses `panphon.distance.feature_edit_distance` — articulatory feature distance as in [Mortensen et al. 2016].
- Handles compound phonemes, long vowels (colons), stress markers, diphthongs.

This same mechanism supports **word-level alignment for unseen languages** (Dutch, German, Hebrew) without any additional training. German and Dutch use MFA dictionaries mapped through this pipeline; Hebrew — where MFA has no dictionary — can only be handled by FDNFA.

**Usage:** set `language = "dutch"` in `solver.py forward()` or `predict.py main_predict()`.

---

## 13. Configuration System

**File:** `conf/config.yaml` — managed by **Hydra** (`@hydra.main(config_path='conf/config.yaml')`).

### Dataset paths
```yaml
timit_path:    /home/rotem/projects/datasets/timit/playground
buckeye_path:  /home/rotem/projects/datasets/buckeye/
data: timit            # 'timit' or 'buckeye'
buckeye_percent: 1.0   # Fraction of Buckeye training data
val_ratio: 0.1         # Fraction of TIMIT train held out for validation
dataloader_n_workers: 10
```

### Model
```yaml
z_dim: 256            # CNN encoder output dim
z_proj: 64            # Projection head output dim
z_proj_linear: true   # Linear vs. MLP projection
z_proj_dropout: 0
latent_dim: 0         # Intermediate CNN dim (0 = use z_dim)
pred_steps: 1         # Contrastive prediction offsets
pred_offset: 0
cosine_coef: 1.0      # Scale factor for cosine similarity
num_classes: 39       # LH39 phoneme set
```

### Optimization
```yaml
optimizer: adamW
lr: 3e-4
lr_anneal_gamma: 0.95
lr_anneal_step: 1000
epochs: 200
batch_size: 8
seed: 100
```

### Run directory
Hydra writes each run into a timestamped subdirectory. Active output dir is set via `hydra.run.dir`. Run directories are under the old DCAF2.0 path in the config comments (historical).

---

## 14. Experiment Tracking

W&B project: `"DCAF2.0"`. All metrics logged via `wandb.log()` in `solver.generic_eval_end()`.

**Logged metrics per epoch:**

| Metric | Description |
|---|---|
| `{mode}_nfc_loss` | Total combined loss |
| `{mode}_ph_loss` | Cross-entropy phoneme classification loss |
| `{mode}_nce_loss` | NCE contrastive loss |
| `{mode}_softDPmse_loss` | Boundary MSE (Soft-DP regression) |
| `{mode}_w_pos` / `{mode}_w_neg` | Learned α (pos/neg weight) |
| `{mode}_w_phi1` / `{mode}_w_phi2` | Learned W₁, W₂ (φ₁/φ₂ decoder weights) |
| `{mode}_min_l1_dist` | Best L1 boundary distance seen so far |
| `current_lr` | Learning rate |

---

## 15. Ablations

From the paper's ablation tables:

### Loss function (MNCE vs. InfoNCE)
| Loss | P@10 | P@25 | P@50 | P@100 |
|---|---|---|---|---|
| **MNCE (ours)** | **39.18** | **83.98** | **95.26** | **98.64** |
| InfoNCE (CPC) | 21.5 | 36.93 | 53.65 | 72.51 |

MNCE outperforms classic InfoNCE by a large margin — hard negative sampling from boundary regions is decisive.

### Decision module (Soft-DP vs. alternatives)
| Decision method | Bi-LSTM | P@10 | P@25 | P@50 | P@100 |
|---|---|---|---|---|---|
| Hard DP | No | 20.9 | 40.2 | 59.05 | 77.79 |
| Hard DP | Yes | 27.9 | 49.06 | 66.11 | 82.38 |
| Naive Peak Detection | No | 24.7 | 62.0 | 98.0 | 99.0 |
| **Soft-DP (ours)** | **Yes** | **39.18** | **83.98** | **95.26** | **98.64** |

Key insight: Soft-DP + Bi-LSTM context = full joint end-to-end training → large accuracy gains.

Ablation implementations in codebase:
- `phoneme_alignment_Hard_DP()` in `utils.py`
- `phoneme_alignment_naive_peak_detection()` in `utils.py`
- `loss_InfoNCE_classic()` in `next_frame_classifier.py`

---

## 16. Key Results

### Phoneme-level (English)
| Model | Dataset | P@10 | P@25 | P@50 | P@100 |
|---|---|---|---|---|---|
| MFA | TIMIT | 38.6 | 72.3 | 81.1 | 84.6 |
| **FDNFA** | **TIMIT** | **39.18** | **83.98** | **95.26** | **98.64** |
| MFA | Buckeye | **35.3** | 60.6 | 68.9 | 72.70 |
| **FDNFA** | **Buckeye** | 29.44 | **68.69** | **88.5** | **96.60** |

### Phoneme-level (unseen languages, zero-shot)
| Language | Model | P@25 | P@50 | P@100 |
|---|---|---|---|---|
| Dutch (IFA) | MFA | 21.80 | 33.90 | 51.02 |
| Dutch (IFA) | **FDNFA** | **38.82** | **55.48** | **73.66** |
| German (PHONDAT) | MFA | 45.83 | 66.78 | 79.19 |
| German (PHONDAT) | **FDNFA** | **47.57** | **69.58** | **83.32** |
| Hebrew | FDNFA only | 27.06 | 41.57 | 59.02 |

### Word-level generalization (no additional training)
FDNFA outperforms or is competitive with MMS and WhisperX on English TIMIT/Buckeye and unseen Dutch/Hebrew, using only phoneme-level training.

---

## 17. Data Formats

### Input audio
- WAV files, **16 kHz mono**.

### Phoneme annotation files (`.phn`)
TIMIT format — one line per phoneme segment:
```
<start_sample>  <end_sample>  <phoneme_label>
```
Example:
```
0     3050  h#
3050  4559  sh
4559  5723  ix
```

### Checkpoint format (`.pt`)
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,      # Keys: NFC.enc.*, NFC.bi_lstm.*, NFC.fc.*,
                                          #       NFC.w_phi, NFC.w_pos_neg
    'optimizer_state_dict': dict,
    'hparams': argparse.Namespace,        # Full config at training time
    'peak_detection_params': bytes        # dill.dumps(defaultdict(...))
}
```

---

## 18. Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `z_dim` | 256 | CNN encoder output dimension D |
| `z_proj` | 64 | Projection head output (BiLSTM input size) |
| `num_classes` | 39 | LH39 phoneme set size |
| `pred_steps` | 1 | Contrastive offset steps |
| `batch_size` | 8 | Per-GPU batch size |
| `lr` | 3e-4 | AdamW learning rate |
| `lr_anneal_gamma` | 0.95 | StepLR decay per step |
| `lr_anneal_step` | 1000 | StepLR step size (batches) |
| `epochs` | 200 | Total training epochs |
| `seed` | 100 | Global random seed |
| `val_ratio` | 0.1 | TIMIT validation fraction |
| `N` (MNCE) | 5 | Positive/negative samples per anchor |
| `δ` (MNCE) | ±1 frame | Boundary sampling half-window |
| `γ` (Soft-DP) | 1e-20 | LogSumExp temperature |
| `η` (CE weight) | 2e-9 | Cross-entropy loss scale |
| `μ` (SoftDP weight) | 1e-4 | Boundary MSE loss scale |
| `devices` | [7] | GPU device index |
| `prominence` | 0.1 | Inference-only peak detection threshold |
