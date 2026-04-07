#!/bin/bash

# Example 4: Train PubCheF-1 locally (no Weights & Biases)
# This runs the default training configuration defined in train.py and writes outputs to:
# PubCheF-1/models/test/

cd "$(dirname "$0")/../PubCheF-1"

# Abort if no CUDA GPU is available — training on CPU is not practical
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU detected. Aborting.'" || exit 1

echo "Starting local PubCheF-1 training run..."

python train.py

# Tip: add --debug for a quick sanity check run on a small subset.
# Example:
# python train.py --debug