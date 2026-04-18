#!/bin/bash
# Quick wrapper script for visualizing latent representations
# Usage: ./visualize_latent.sh <audio_file> <run_directory>

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <audio_file> <run_directory> [prominence] [w_phi]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/audio.wav /path/to/run/directory"
    echo ""
    echo "Optional arguments:"
    echo "  prominence (default: 0.05)"
    echo "  w_phi (default: 0.5)"
    exit 1
fi

WAV_FILE="$1"
RUN_DIR="$2"
PROMINENCE="${3:-0.05}"
W_PHI="${4:-0.5}"

# Validate inputs
if [ ! -f "$WAV_FILE" ]; then
    echo "ERROR: Audio file not found: $WAV_FILE"
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory not found: $RUN_DIR"
    exit 1
fi

echo "========================================"
echo "Visualizing Latent Representation"
echo "========================================"
echo "Audio file: $WAV_FILE"
echo "Run directory: $RUN_DIR"
echo "Prominence: $PROMINENCE"
echo "W_phi: $W_PHI"
echo "========================================"
echo ""

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run the visualization
python visualize_latent_representation.py \
    --wav "$WAV_FILE" \
    --run-dir "$RUN_DIR" \
    --prominence "$PROMINENCE" \
    --w-phi "$W_PHI"

echo ""
echo "✓ Done! Check the plots in: $RUN_DIR/latent_representations/"
