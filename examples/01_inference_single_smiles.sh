#!/bin/bash

# Example 1: Inference on a single SMILES string using PubCheF-1 ensemble model
# This doesn't use p-value thresholding, so it outputs a CSV with all prediction probabilities.

# Ensure we're running from the root of the project
cd "$(dirname "$0")/../PubCheF-1"

# Abort if no CUDA GPU is available
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU detected. Aborting.'" || exit 1

echo "Running Inference on a single SMILES string..."

python inference.py \
    --smiles "CCO" \
    --compound_id "Ethanol" \
    --model_name "ensemble_single_canon_chiral_20epoch" \
    --mlb_dir "../data/final_datasets/preprocessed_propagated_1_hard"
