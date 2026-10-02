#!/bin/bash

# Example 3: Extract Embeddings on a full CSV
# This will extract the raw ChemBERTa 384-D embeddings from the [CLS] token and bypass prediction layers.
# The embeddings will be saved to PubCheF-1/inference_results/<model_name>/embeddings/<csv_name>/
# as a .pt dict with keys 'metadata' (input CSV rows) and 'embeddings' (tensor of shape [n_molecules, 384]).
# Uses the bundled 3-molecule file PubCheF-1/data/sample_mols.csv.

cd "$(dirname "$0")/../PubCheF-1"

# Runs on CPU if no CUDA GPU is available
python -c "import torch; torch.cuda.is_available() or print('WARNING: No CUDA GPU detected. Running on CPU.')"

echo "Extracting Embeddings from a CSV..."

export CUDA_VISIBLE_DEVICES=0  # Use the first GPU
python inference.py \
    --input_csv "data/sample_mols.csv" \
    --smiles_column "smiles" \
    --extract_embeddings \
    --batch_size 100 \
    --model_name "exalted-sweep-1" \
    --mlb_dir "../data/final_datasets/preprocessed_propagated_1_hard"
