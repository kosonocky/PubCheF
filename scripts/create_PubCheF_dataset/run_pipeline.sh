#!/bin/bash
# PubCheF dataset creation pipeline.
#
# Before running, export your API credentials:
#   export OPENAI_API_KEY="sk-..."
#   export ENTREZ_EMAIL="you@example.com"
#   export ENTREZ_API_KEY="..."   # optional, increases NCBI rate limit
#
# Then: bash run_pipeline.sh
set -euo pipefail

# USER VARS
# PUBCHEM_DIR="../../data/pubchem/$(date +%Y%m%d)" # fresh download to today's date
PUBCHEM_DIR="../../data/pubchem/20260328"       # reuse existing data (step 01 skips download if files already exist)
GPT_MODEL="gpt-5.4-mini"                  # step 02: model to use for PMID annotation. We used a custom fine-tuned model in the manuscript, but we set this to a base model for reproducibility. You can also specify a custom fine-tuned model here if you have one.
EMBEDDING_MODEL="text-embedding-3-large"  # step 03: model to use for label embedding
EPS="0.41"      # DBSCAN epsilon — steps 04 and 05 must use the same value. 0.41 was the optimal value found in the paper
SCAN_EPS=false  # set to true to sweep eps 0.2→0.8 (slow, only use for development)
N_PMIDS="100" # Set to e.g. 200 for a quick demo; leave empty to process all PMIDs



# ──────────────────────────────────────────────────────────────────────────────
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-..."; exit 1
fi
if [[ -z "${ENTREZ_EMAIL:-}" ]]; then
    echo "ERROR: ENTREZ_EMAIL is not set. Run: export ENTREZ_EMAIL=you@example.com"; exit 1
fi

run_step() {
    local step="$1"; local script="$2"; shift 2
    echo ""; echo "=== Step $step: $script ==="
    python "$script" "$@"
}

N_PMIDS_ARGS=();  [[ -n "$N_PMIDS" ]] && N_PMIDS_ARGS=(--n_pmids "$N_PMIDS")
SCAN_EPS_ARGS=(); $SCAN_EPS          && SCAN_EPS_ARGS=(--scan_eps)

# run_step 1 01_load_pubchem_files.py         --pubchem_dir "$PUBCHEM_DIR"
# run_step 2 02_pmid_to_func.py               --gpt_model "$GPT_MODEL" --pubchem_dir "$PUBCHEM_DIR" "${N_PMIDS_ARGS[@]}"
# run_step 3 03_embed_labels.py               --gpt_model "$GPT_MODEL" --embedding_model "$EMBEDDING_MODEL"
# run_step 4 04_cluster_labels_dbscan.py      --gpt_model "$GPT_MODEL" --eps "$EPS" "${SCAN_EPS_ARGS[@]}"
# run_step 5 05_map_to_pmid_dataset.py        --gpt_model "$GPT_MODEL" --eps "$EPS" --pubchem_dir "$PUBCHEM_DIR"
run_step 6 06_create_smiles_func_dataset.py --gpt_model "$GPT_MODEL" --pubchem_dir "$PUBCHEM_DIR"
run_step 7 07_label_propagation.py
run_step 8 08_preprocess_data_label_prop.py

echo ""; echo "=== Pipeline complete ==="
