#!/bin/bash

# Example 2: Inference on a full CSV with p-value thresholding using PubCheF-1 ensemble model
# This uses --p_threshold to save only significant predictions, lowering output size dramatically.
# The predictions will be saved to PubCheF-1/inference_results/<model_name>/predictions/<csv_name>/
# Each output row is the input row plus a `top_preds` column: a {label: probability} dict of labels above the threshold.

# Uses the bundled 3-molecule file PubCheF-1/data/sample_mols.csv (ethanol, acetic acid, benzene).
# To run on your own molecules, point --input_csv at any CSV and set --smiles_column to its SMILES column.

cd "$(dirname "$0")/../PubCheF-1"

# Runs on CPU if no CUDA GPU is available
python -c "import torch; torch.cuda.is_available() or print('WARNING: No CUDA GPU detected. Running on CPU.')"

echo "Running Inference on a CSV predicting significant terms..."

export CUDA_VISIBLE_DEVICES=0  # Use the first GPU
python inference.py \
    --input_csv "data/sample_mols.csv" \
    --smiles_column "smiles" \
    --p_threshold 0.05 \
    --batch_size 100 \
    --model_name "ensemble_single_canon_chiral_20epoch" \
    --mlb_dir "../data/final_datasets/preprocessed_propagated_1_hard"
