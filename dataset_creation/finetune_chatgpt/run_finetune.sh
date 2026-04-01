#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set." >&2
    exit 1
fi

echo "=== Step 1: Aggregate fine-tune examples ==="
python 01_aggregate_for_finetune_examples.py

echo ""
echo "=== Step 2: Upload and fine-tune ==="
python 02_fine_tune.py
