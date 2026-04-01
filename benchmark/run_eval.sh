#!/usr/bin/env bash
# Run the full GPT evaluation pipeline for all models × benchmarks.
#
# Input files must be in the format:
# dataset must have cols: id, smiles, ground_truth
# predictions must have cols: id, smiles, prediction
#
# Usage:
#   ./run_eval.sh [--gpt-model MODEL]
#
# Output:
#   results/gpt_eval/<model>_<benchmark>/   per-run chunk CSVs
#   results/scores.txt                      final precision/recall/F1 table
#
# Before running, export your API credentials:
#   export OPENAI_API_KEY="sk-..."

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$BENCHMARK_DIR/scripts"
PREDS="$BENCHMARK_DIR/predictions"
DATASET="$BENCHMARK_DIR/dataset"
EVAL_DIR="$BENCHMARK_DIR/results/gpt_eval"
SCORES="$BENCHMARK_DIR/results/scores.txt"

# GPT_MODEL="gpt-4-turbo-2024-04-09"
GPT_MODEL="gpt-5.4-mini"
echo "=== GPT model: $GPT_MODEL ==="

# ------------------------------------------------------------------
# GPT evaluation — one run per model × benchmark combination
# ------------------------------------------------------------------

# # chef-fplr | PubCheF
python "$SCRIPTS/gpt_eval.py" \
    --input      "$PREDS/pubchef-test-fpbal-0.5_chef-fplr-above-testDS-cutoff.csv" \
    --dataset    "$DATASET/pubchef-test-fpbal-0.5.csv" \
    --model-type labels \
    --gpt-model  "$GPT_MODEL" \
    --output-dir "$EVAL_DIR/chef-fplr_pubchef"

# chef-fplr | OpenTargets
python "$SCRIPTS/gpt_eval.py" \
    --input      "$PREDS/opentargets-20240626_chef-fplr-above-testDS-cutoff.csv" \
    --dataset    "$DATASET/opentargets-20240626.csv" \
    --model-type labels \
    --gpt-model  "$GPT_MODEL" \
    --output-dir "$EVAL_DIR/chef-fplr_opentargets"

# pubchef-1 | PubCheF
python "$SCRIPTS/gpt_eval.py" \
    --input      "$PREDS/pubchef-test-fpbal-0.5_pubchef-1-above-valDS-cutoff.csv" \
    --dataset    "$DATASET/pubchef-test-fpbal-0.5.csv" \
    --model-type labels \
    --gpt-model  "$GPT_MODEL" \
    --output-dir "$EVAL_DIR/pubchef-1_pubchef"

# pubchef-1 | OpenTargets
python "$SCRIPTS/gpt_eval.py" \
    --input      "$PREDS/opentargets-20240626_pubchef-1-above-valDS-cutoff.csv" \
    --dataset    "$DATASET/opentargets-20240626.csv" \
    --model-type labels \
    --gpt-model  "$GPT_MODEL" \
    --output-dir "$EVAL_DIR/pubchef-1_opentargets"

# molt5 | PubCheF
python "$SCRIPTS/gpt_eval.py" \
    --input      "$PREDS/pubchef-test-fpbal-0.5_molt5.csv" \
    --dataset    "$DATASET/pubchef-test-fpbal-0.5.csv" \
    --model-type free-text \
    --gpt-model  "$GPT_MODEL" \
    --output-dir "$EVAL_DIR/molt5_pubchef"

# molt5 | OpenTargets
python "$SCRIPTS/gpt_eval.py" \
    --input      "$PREDS/opentargets-20240626_molt5.csv" \
    --dataset    "$DATASET/opentargets-20240626.csv" \
    --model-type free-text \
    --gpt-model  "$GPT_MODEL" \
    --output-dir "$EVAL_DIR/molt5_opentargets"

# ------------------------------------------------------------------
# Compute scores
# ------------------------------------------------------------------

python "$SCRIPTS/process_eval_results.py" \
    --eval "chef-fplr (PubCheF):$EVAL_DIR/chef-fplr_pubchef" \
    --eval "chef-fplr (OpenTargets):$EVAL_DIR/chef-fplr_opentargets" \
    --eval "pubchef-1 (PubCheF):$EVAL_DIR/pubchef-1_pubchef" \
    --eval "pubchef-1 (OpenTargets):$EVAL_DIR/pubchef-1_opentargets" \
    --eval "molt5 (PubCheF):$EVAL_DIR/molt5_pubchef" \
    --eval "molt5 (OpenTargets):$EVAL_DIR/molt5_opentargets" \
    --output "$SCORES"
