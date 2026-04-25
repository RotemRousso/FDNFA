"""
Inference script to visualize CNN latent representations
Compares epoch 0 (untrained) vs best trained epoch
Saves heatmaps showing how the CNN learns features across frames
"""

import os
import argparse
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from argparse import Namespace
import hydra
from omegaconf import DictConfig, OmegaConf
import dill

from next_frame_classifier import NextFrameClassifier
from dataloader import spectral_size


def load_checkpoint(ckpt_path, model, device, verbose=True):
    """Load checkpoint into model."""
    if not os.path.exists(ckpt_path):
        if verbose:
            print(f"WARNING: Checkpoint not found at {ckpt_path}")
        return False
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in ckpt:
        weights = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        weights = ckpt['model_state_dict']
    else:
        weights = ckpt
    
    # Remove 'NFC.' prefix if present (from Solver wrapper)
    weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
    
    try:
        model.load_state_dict(weights)
        if verbose:
            print(f"✓ Loaded checkpoint: {ckpt_path}")
        return True
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return False


def extract_latent_representation(model, audio, device):
    """
    Extract CNN latent representation (z) for all frames.
    Returns normalized latent features: (num_frames, z_dim)
    """
    model.eval()
    audio = audio.unsqueeze(0).unsqueeze(1).to(device)  # (1, 1, audio_len)
    
    with torch.no_grad():
        # Forward through encoder (stops before LSTM)
        # The encoder outputs (batch, frames, z_dim) after transpose
        z = model.enc(audio)  # (1, frames, z_dim)
        z = F.normalize(z, dim=-1)  # normalize
        z = z.squeeze(0).detach().cpu().numpy()  # (frames, z_dim)
    
    return z


def create_latent_heatmap(latent_repr, title, output_path, cmap='viridis'):
    """
    Create and save a heatmap of latent representation.
    Similar to a spectrogram but shows CNN feature activations.
    
    Args:
        latent_repr: (num_frames, z_dim) array
        title: plot title
        output_path: where to save the figure
        cmap: colormap
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Transpose so frames are on x-axis, features on y-axis
    im = ax.imshow(latent_repr.T, aspect='auto', cmap=cmap, interpolation='nearest')
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Latent Feature Dimension')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Feature Activation')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_comparison_plot(z_epoch0, z_best, output_dir):
    """Create side-by-side comparison plots."""
    
    # Normalize both to same scale for fair comparison
    vmin_0 = z_epoch0.min()
    vmax_0 = z_epoch0.max()
    vmin_best = z_best.min()
    vmax_best = z_best.max()
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Epoch 0
    im0 = axes[0].imshow(z_epoch0.T, aspect='auto', cmap='viridis', 
                         vmin=min(vmin_0, vmin_best), vmax=max(vmax_0, vmax_best),
                         interpolation='nearest')
    axes[0].set_xlabel('Frame Index')
    axes[0].set_ylabel('Latent Feature Dimension')
    axes[0].set_title('CNN Latent Representation - Epoch 0 (Untrained)', 
                      fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], label='Feature Activation')
    
    # Best epoch
    im_best = axes[1].imshow(z_best.T, aspect='auto', cmap='viridis',
                             vmin=min(vmin_0, vmin_best), vmax=max(vmax_0, vmax_best),
                             interpolation='nearest')
    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Latent Feature Dimension')
    axes[1].set_title('CNN Latent Representation - Best Trained Epoch', 
                      fontsize=12, fontweight='bold')
    plt.colorbar(im_best, ax=axes[1], label='Feature Activation')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'latent_comparison_epoch0_vs_best.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {comparison_path}")
    plt.close()


def create_difference_plot(z_epoch0, z_best, output_dir):
    """Create a plot showing the difference between epoch 0 and best."""
    diff = z_best - z_epoch0
    
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(diff.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Latent Feature Dimension')
    ax.set_title('CNN Latent Representation Difference (Best - Epoch0)', 
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Change in Feature Activation')
    plt.tight_layout()
    
    diff_path = os.path.join(output_dir, 'latent_difference_best_minus_epoch0.png')
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {diff_path}")
    plt.close()


def main_visualize(wav_path, run_dir, prominence=None, w_phi=0.5):
    """
    Main inference function to visualize latent representations.
    
    Args:
        wav_path: path to audio file
        run_dir: path to training run directory (contains checkpoints and config)
        prominence: peak detection prominence (not used in this script but kept for API consistency)
        w_phi: weight for phi (not used in this script but kept for API consistency)
    """
    
    print(f"\n{'='*80}")
    print(f"Visualizing CNN Latent Representation")
    print(f"{'='*80}")
    print(f"Audio file: {wav_path}")
    print(f"Run directory: {run_dir}")
    
    # Create output directory for plots
    output_dir = os.path.join(run_dir, 'latent_representations')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============ Load config ============
    config_path = os.path.join(run_dir, '.hydra', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at {config_path}")
        print(f"Trying alternate locations...")
        # Try to find config in parent directories
        for parent in [run_dir, os.path.dirname(run_dir), os.path.dirname(os.path.dirname(run_dir))]:
            alt_config = os.path.join(parent, 'config.yaml')
            if os.path.exists(alt_config):
                config_path = alt_config
                break
    
    try:
        cfg = OmegaConf.load(config_path)
        print(f"✓ Loaded config from {config_path}")
    except Exception as e:
        print(f"ERROR loading config: {e}")
        print(f"Using default hyperparameters...")
        # Create a minimal config with defaults
        cfg = OmegaConf.create({
            'z_dim': 256,
            'z_proj': 64,
            'z_proj_linear': True,
            'z_proj_dropout': 0,
            'latent_dim': 0,
            'pred_steps': 1,
            'pred_offset': 0,
            'num_classes': 39,
            'cosine_coef': 1.0
        })
    
    # Convert to Namespace for NextFrameClassifier compatibility
    if isinstance(cfg, DictConfig):
        cfg = Namespace(**OmegaConf.to_container(cfg, resolve=True))
    
    # ============ Load audio ============
    print(f"\nLoading audio...")
    audio, sr = torchaudio.load(wav_path)
    if sr != 16000:
        print(f"Resampling from {sr} to 16000 Hz")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
        sr = 16000
    
    audio = audio[0]  # remove channel dimension
    print(f"✓ Audio shape: {audio.shape}, Sample rate: {sr}")
    
    # ============ Create model ============
    print(f"\nCreating model...")
    model = NextFrameClassifier(cfg).to(device)
    print(f"✓ Model created")
    print(f"  - Z dimension: {cfg.z_dim}")
    print(f"  - Z projection: {cfg.z_proj}")
    print(f"  - Latent dimension: {cfg.latent_dim}")
    
    # ============ Try to load epoch 0 checkpoint ============
    print(f"\n{'='*80}")
    print(f"Loading Epoch 0 (Initial) Checkpoint")
    print(f"{'='*80}")
    
    epoch0_ckpt = os.path.join(run_dir, '0_best_model.pt')
    z_epoch0 = None
    
    if load_checkpoint(epoch0_ckpt, model, device, verbose=True):
        z_epoch0 = extract_latent_representation(model, audio, device)
        print(f"✓ Extracted latent representation")
        print(f"  - Shape: {z_epoch0.shape}")
        print(f"  - Min: {z_epoch0.min():.4f}, Max: {z_epoch0.max():.4f}, Mean: {z_epoch0.mean():.4f}")
        
        # Save individual plot for epoch 0
        epoch0_path = os.path.join(output_dir, 'latent_epoch0_untrained.png')
        create_latent_heatmap(z_epoch0, 'CNN Latent Representation - Epoch 0 (Untrained)', epoch0_path)
    else:
        print(f"⚠ Could not load epoch 0 checkpoint, will use random initialization")
        z_epoch0 = extract_latent_representation(model, audio, device)
        print(f"✓ Extracted latent representation from random initialization")
        print(f"  - Shape: {z_epoch0.shape}")
        
        epoch0_path = os.path.join(output_dir, 'latent_random_init.png')
        create_latent_heatmap(z_epoch0, 'CNN Latent Representation - Random Initialization', epoch0_path)
    
    # ============ Load best checkpoint ============
    print(f"\n{'='*80}")
    print(f"Loading Best Trained Checkpoint")
    print(f"{'='*80}")
    
    # Find best checkpoint (look for the latest checkpoint)
    best_ckpt_candidates = []
    for f in os.listdir(run_dir):
        if f.endswith('_best_model.pt') and f != '0_best_model.pt':
            best_ckpt_candidates.append(os.path.join(run_dir, f))
    
    if best_ckpt_candidates:
        # Sort by modification time and get the latest
        best_ckpt = sorted(best_ckpt_candidates, key=os.path.getmtime, reverse=True)[0]
        print(f"Found checkpoint: {os.path.basename(best_ckpt)}")
    else:
        best_ckpt = os.path.join(run_dir, 'best_model.pt')
        print(f"Looking for: best_model.pt")
    
    z_best = None
    if load_checkpoint(best_ckpt, model, device, verbose=True):
        z_best = extract_latent_representation(model, audio, device)
        print(f"✓ Extracted latent representation")
        print(f"  - Shape: {z_best.shape}")
        print(f"  - Min: {z_best.min():.4f}, Max: {z_best.max():.4f}, Mean: {z_best.mean():.4f}")
        
        # Save individual plot for best epoch
        best_path = os.path.join(output_dir, 'latent_best_trained.png')
        create_latent_heatmap(z_best, 'CNN Latent Representation - Best Trained Epoch', best_path)
    else:
        print(f"⚠ Could not load best checkpoint")
        z_best = z_epoch0  # fall back to epoch 0
    
    # ============ Create comparison plots ============
    if z_epoch0 is not None and z_best is not None:
        print(f"\n{'='*80}")
        print(f"Creating Comparison Plots")
        print(f"{'='*80}")
        
        create_comparison_plot(z_epoch0, z_best, output_dir)
        create_difference_plot(z_epoch0, z_best, output_dir)
        
        print(f"\n✓ Comparison analysis:")
        print(f"  - Mean activation change: {np.mean(z_best - z_epoch0):.4f}")
        print(f"  - Std of activation change: {np.std(z_best - z_epoch0):.4f}")
    
    print(f"\n{'='*80}")
    print(f"✓ All plots saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize CNN latent representations at epoch 0 vs best trained epoch'
    )
    parser.add_argument('--wav', type=str, required=True,
                        help='Path to audio file (.wav)')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to training run directory (where checkpoints are saved)')
    parser.add_argument('--prominence', type=float, default=0.05,
                        help='Peak detection prominence (for API consistency, not used)')
    parser.add_argument('--w-phi', type=float, default=0.5,
                        help='Weight for phi (for API consistency, not used)')
    
    args = parser.parse_args()
    
    main_visualize(
        wav_path=args.wav,
        run_dir=args.run_dir,
        prominence=args.prominence,
        w_phi=args.w_phi
    )
