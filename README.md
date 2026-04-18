# FDNFA

Fully Differentiable Neural Forced Alignment (FDNFA) is an end-to-end architecture for phoneme alignment designed to close the performance gap between modern ASR and traditional forced-alignment systems.

## Overview

Recent sequence-modeling advances have greatly improved ASR robustness and accuracy, while forced alignment still commonly relies on strong but older HMM-GMM pipelines.  
FDNFA replaces that setup with a fully differentiable neural framework optimized jointly from input signal to alignment output.

## Method

The model combines:

- **Encoder** with two complementary branches:
  - a **phoneme identity verification** branch
  - a **phoneme boundary detection** branch
- **Decoder** implemented as a trainable **differentiable soft dynamic programming** module that produces alignment decisions
- A novel **contrastive loss** that separates steady-state phoneme regions from transition boundaries

The full system is trained end-to-end.

## Results

FDNFA:

- Outperforms previous state of the art on hand-annotated English phoneme-alignment benchmarks
- Shows strong generalization to word-level alignment
- Demonstrates transfer to unseen languages
