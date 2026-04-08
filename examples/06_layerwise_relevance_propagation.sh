#!/bin/bash

# Example 6: LRP token importance for a single SMILES string
# Generates an atom-colored SVG showing which tokens the model attends to
# for a given label prediction, using Layer-wise Relevance Propagation (LRP).
#
# Outputs saved to PubCheF-1/figs/:
#   <name>_<label>.svg          — molecule with per-atom importance coloring
#   <label>_colorbar.png        — color scale for the importance values

# Must run from Transformer-Explainability/ so that BERT_explainability imports resolve
cd "$(dirname "$0")/../PubCheF-1/Transformer-Explainability"

# Abort if no CUDA GPU is available
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU detected. Aborting.'" || exit 1

python layerwise_relevance_propagation.py \
    --smiles "CN1CC(=O)OB(c2csc(C=O)c2)OC(=O)C1" \
    --name "62" \
    --label_to_explain "Beta-Lactamase Inhibitor" \
    --model_name "ensemble_single_canon_chiral_20epoch" \
