#!/bin/bash

# Example 2: Inference on a full CSV with p-value thresholding using PubCheF-1 ensemble model
# This uses --p_threshold to save only significant predictions, lowering output size dramatically.
# The predictions will be saved to inference_results/<model_type>/predictions/<csv_name>/

# For this example, we assume you have a file `data/sample_mols.csv` containing a SMILES column.

cd "$(dirname "$0")/../PubCheF-1"

# Abort if no CUDA GPU is available
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU detected. Aborting.'" || exit 1

# Create a small dummy CSV to test
# mkdir -p data
# echo "smiles,cid,notes" > data/sample_mols.csv
# echo "CCO,101,Ethanol" >> data/sample_mols.csv
# echo "CC(=O)O,102,Acetic Acid" >> data/sample_mols.csv
# echo "C1=CC=CC=C1,103,Benzene" >> data/sample_mols.csv

echo "Running Inference on a CSV predicting significant terms..."

export CUDA_VISIBLE_DEVICES=0  # Use the first GPU
python inference.py \
    --input_csv "data/20260427_nicholas_page2.csv" \
    --smiles_column "smiles" \
    --p_threshold 0.1 \
    --batch_size 10000 \
    --model_name "ensemble_single_canon_chiral_20epoch" \
    --mlb_dir "../data/final_datasets/preprocessed_propagated_1_hard"
