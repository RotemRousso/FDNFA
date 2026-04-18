# Latent Representation Visualization

This script visualizes how the CNN encoder learns features by comparing the latent representation at epoch 0 (untrained) vs the best trained epoch.

## What it does

1. **Extracts CNN latent features** from the encoder for all frames of an audio file
2. **Visualizes as heatmaps** (like spectrograms but showing learned CNN features instead of raw audio spectral content)
3. **Compares epoch 0 vs best epoch** to show how the model learns
4. **Saves 4 plots**:
   - `latent_epoch0_untrained.png` - Features at epoch 0
   - `latent_best_trained.png` - Features at best epoch
   - `latent_comparison_epoch0_vs_best.png` - Side-by-side comparison
   - `latent_difference_best_minus_epoch0.png` - Difference map (red = improved, blue = weakened)

## Usage

### Basic usage

```bash
cd /home/rotem/projects/CFA/DCAF2/DCAF2.0
python visualize_latent_representation.py \
  --wav <path_to_audio.wav> \
  --run-dir <path_to_training_run_directory>
```

### Example

Assuming you trained a model and checkpoints are in `/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyeOrig_tmux0Elektra_gpu0/YYYY-MM-DD_HH-MM-SS-default`:

```bash
python visualize_latent_representation.py \
  --wav /path/to/your/audio.wav \
  --run-dir /home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyeOrig_tmux0Elektra_gpu0/YYYY-MM-DD_HH-MM-SS-default
```

### Optional parameters

```bash
python visualize_latent_representation.py \
  --wav /path/to/audio.wav \
  --run-dir /path/to/run \
  --prominence 0.05 \
  --w-phi 0.5
```

- `--prominence`: Peak detection threshold (not used in this script, kept for API consistency)
- `--w-phi`: Weight for phi (not used in this script, kept for API consistency)

## Output

Plots are saved to: `<run_dir>/latent_representations/`

## How to find your run directory

After training with `main.py`, look for the run directory path in the training output. It's typically:

```
/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/<run_name>/<YYYY-MM-DD_HH-MM-SS>-default/
```

For example:
```
/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyeOrig_tmux0Elektra_gpu0/2025-12-02_14-30-15-default/
```

## What to look for in the plots

1. **Epoch 0 plot**: Should show relatively uniform/random activation patterns (CNN hasn't learned yet)
2. **Best trained plot**: Should show structured patterns - distinct activations for different parts of the audio
3. **Difference plot**: 
   - **Red regions**: Features that became stronger with training
   - **Blue regions**: Features that became weaker with training
   - Should show clear changes indicating learning occurred

## Troubleshooting

### "Config not found"
The script will try multiple locations. Make sure your run directory structure is intact from training.

### "Could not load epoch 0 checkpoint"
This is OK - the script will use random initialization instead. The comparison will show the epoch 0 with random weights vs. best trained weights.

### "Could not load best checkpoint"
Check that checkpoint files exist in your run directory with names like:
- `0_best_model.pt` (epoch 0)
- `1_best_model.pt`, `2_best_model.pt`, etc. (other epochs)
- `best_model.pt` (best overall)

### Memory issues
If you get CUDA OOM errors, add to your run command:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python visualize_latent_representation.py --wav ... --run-dir ...
```

Or reduce batch size in the config.

## Integration with predict.py

You can use this script independently or alongside `predict.py`. Both scripts:
- Load the same checkpoint format
- Work with the same model architecture
- Can be run on the same audio files

This script focuses specifically on understanding CNN feature learning, while `predict.py` focuses on final predictions and peak detection.
