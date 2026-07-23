#!/bin/bash

# Example 3: Extract Embeddings on a full CSV
# This will extract the raw ChemBERTa 768-D embeddings from the [CLS] token and bypass prediction layers.
# The embeddings will be saved to inference_results/<model_type>/embeddings/<csv_name>/

cd "$(dirname "$0")/../PubCheF-1"

# Abort if no CUDA GPU is available
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU detected. Aborting.'" || exit 1

# Create a small dummy CSV to test
mkdir -p data
echo "smiles,cid,notes" > data/sample_mols.csv
echo "CCO,101,Ethanol" >> data/sample_mols.csv
echo "CC(=O)O,102,Acetic Acid" >> data/sample_mols.csv
echo "C1=CC=CC=C1,103,Benzene" >> data/sample_mols.csv

echo "Extracting Embeddings from a CSV..."

export CUDA_VISIBLE_DEVICES=0  # Use the first GPU
python inference.py \
    --input_csv "data/sample_mols.csv" \
    --smiles_column "smiles" \
    --extract_embeddings \
    --batch_size 100 \
    --model_name "exalted-sweep-1" \
    --mlb_dir "../data/final_datasets/preprocessed_propagated_1_hard"
