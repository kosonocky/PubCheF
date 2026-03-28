#!/bin/bash

# Example 5: Evaluate the PubCheF-1 ensemble on the test set
# Computes ROC-AUC, PR-AUC, and Brier score per label and mean across all labels.
# Also finds F1-optimal probability thresholds per label.
#
# Outputs saved to models/<model_name>/:
#   scaffold_split_test_ensemble_metrics_indiv.csv     — per-label metrics
#   scaffold_split_test_ensemble_metrics_mean.txt      — mean metrics
#   scaffold_split_test_ensemble_f1optimal_cutoffs.csv — per-label F1-optimal thresholds
#   scaffold_split_test_ensemble_predictions.csv       — full probability matrix

cd "$(dirname "$0")/../PubCheF-1"

echo "Evaluating ensemble on test set..."

python eval.py \
    --model_name "ensemble_single_canon_chiral_20epoch" \
    --data_dir "../data/final_datasets/preprocessed_propagated_1_hard" \
    --split_type "scaffold_split" \
    --splits "test" \
    --batch_size 128

# To also evaluate on val:
#   --splits "val,test"
#
# For a quick smoke test on the first 1000 rows:
#   --debug
