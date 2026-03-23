# PubCheF

## Setup Environment

If you want to use the ML models here, you can easily set up the Python environment using `uv`:

```bash
# Assuming you have UV installed 
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Running Inference
You can see some template execution scripts inside `examples/`. These illustrate three main modes:
1. `01_inference_single_smiles.sh` - Predict against a single unbatched SMILES query.
2. `02_inference_csv_with_threshold.sh` - Process bulk SMILES entries via CSV with predictions dynamically truncated if probability falls beneath a user-defined threshold.
3. `03_extract_embeddings_csv.sh` - Extract core ChemBERTa representation embeddings natively via CSV.
