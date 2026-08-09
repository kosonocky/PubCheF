# PubCheF
## Manuscript information
Title: Learning from human and chemical languages to predict biological function

Authors: Clayton W. Kosonocky†, Nikol Kadeřábková†, Kangsan Kim†, Ayesha J.S. Mahmood†, Alexander Dunmyre, Phillip Woolley, Kristi Xing, Daniel Winkler, Tomer Babu, Filip Kadeřábek, Jonathan L. Sessler, Eric V. Anslyn, Edward M. Marcotte*, Y. Jessie Zhang*, Andrew D. Ellington*, Despoina A.I. Mavridou*

†These authors have contributed equally to this work

*Correspondence: marcotte@utexas.edu; jzhang@cm.utexas.edu; ellingtonlab@gmail.com; despoina.mavridou@austin.utexas.edu

[URL coming soon]

---

## Reposistory structure:
This repository contains the following:
- **The PubCheF dataset** — ~1.25M molecules (CID, SMILES, PMIDs, functional labels) derived from PubChem bioassay literature, with 5,215 functional labels. Also includes scripts to re-create this dataset.
- **PubCheF-1 model** — a fine-tuned multi-label classifier trained to predict chemical function from structure. Includes scripts to train, inference, validate, and explain.
- **Benchmark for chemical function prediction** — Two benchmark datasets to evaluate chemical function prediction (PubCheF-test and OpenTargets Indications). Includes scripts to run the GPT-assisted evaluation to evaluate models that output free-text or labels.

---

## Hardware Requirements

The modeling scripts (training, inference, explainability) require an NVIDIA CUDA-capable GPU. Dataset creation and benchmark evaluation scripts do not have any hardware requirements.

---

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync
source .venv/bin/activate
```

---

## Repository Structure

```
PubCheF/
├── data/                        # Final dataset files
├── dataset_creation/            # Scripts to reproduce the dataset from scratch. Will overwrite the existing dataset
├── PubCheF-1/                   # Model code (inference, training, eval, explainability)
├── benchmark/                   # Benchmark datasets, predictions, and evaluation pipeline
└── examples/                    # Example shell scripts
```

---

## Dataset

The final PubCheF dataset used in the manuscript can be found in [data/final_datasets/](data/final_datasets/):

| File | Description |
|---|---|
| `cid_smiles_pmid_func_pre_prop.csv` | Raw dataset before label propagation (~1.25M rows). Columns: `cid`, `smiles`, `pmids`, `labels` |
| `cid_smiles_pmid_func_propagated_cutoff_1_hard.csv` | Dataset after label propagation with cutoff=1 (hard labels) |
| `labels.txt` | All 5,215 functional labels with their occurrence counts |
| `preprocessed_propagated_1_hard/` | Preprocessed, split-ready data for model training/eval (includes `mlb.pkl` and scaffold splits) |

---

## Dataset Creation

Scripts to reproduce the dataset from PubChem raw files can be found in [dataset_creation/create_PubCheF_dataset/](dataset_creation/create_PubCheF_dataset/). Run them in order via the pipeline script:

```bash
cd dataset_creation/create_PubCheF_dataset
bash run_pipeline.sh
```

The numbered steps are:

| Script | Description |
|---|---|
| `01_load_pubchem_files.py` | Load and parse raw PubChem compound/bioassay files |
| `02_pmid_to_func.py` | Map PMIDs to functional annotations using a fine-tuned ChatGPT model |
| `03_embed_labels.py` | Embed functional label strings |
| `04_cluster_labels_dbscan.py` | Cluster label embeddings with DBSCAN to merge semantic duplicates |
| `05_map_to_pmid_dataset.py` | Map clustered labels back to the PMID dataset |
| `06_create_smiles_func_dataset.py` | Create compound–function dataset |
| `07_label_propagation.py` | Propagate labels across the compound graph |
| `08_preprocess_data_label_prop.py` | Tokenize and preprocess for model training |

The scripts to fine-tune the GPT model used for step 02 can be found in [dataset_creation/finetune_chatgpt/](dataset_creation/finetune_chatgpt/).

**NOTE**: Building this dataset requires an OPENAI_API_KEY and costs money to run. This cost us a few hundred dollars when it was done in 2024 using a fine-tuned verison of GPT-3.5-turbo. We recommend looking into the current model offerings to decide which model to use and if you want to use a fine-tuned model or not.

---

## PubCheF-1 Model

All model code is in [PubCheF-1/](PubCheF-1/). The key scripts are:

| Script | Purpose |
|---|---|
| `inference.py` | Run predictions on a single SMILES or a CSV |
| `train.py` | Fine-tune a new model |
| `eval.py` | Evaluate model and compute per-label metrics + F1-optimal thresholds |

Trained model weights live in `PubCheF-1/models/`. The primary ensemble is `ensemble_single_canon_chiral_20epoch`.

> **Note:** Model weights are stored via Git LFS. After cloning, run `git lfs pull` to download them before running inference or evaluation. If Git LFS is not installed, run `git lfs install` first.

---

## Examples

We recommend trying out the examples in [examples/](examples/) to understand the syntax of running the model:

| Script | What it does |
|---|---|
| `01_inference_single_smiles.sh` | Predict functional labels for a single SMILES (e.g. `CCO` / Ethanol) |
| `02_inference_csv_with_threshold.sh` | Bulk inference from a CSV with a probability threshold to filter out low-confidence predictions |
| `03_extract_embeddings_csv.sh` | Extract embeddings from SMILES in a CSV |
| `04_train_model.sh` | Launch a local training run |
| `05_evaluate_ensemble_model.sh` | Evaluate ensemble on test/val sets; saves per-label ROC-AUC, PR-AUC, Brier scores, and F1-optimal cutoffs |
| `06_explainability_lrp.sh` | Generate LRP atom-importance for a given SMILES and label |

---

## Benchmark

This benchmark uses an LLM to evaluate chemical function prediction regardless of the output format, allowing for the comparison of GPT-style models with classifier models. The LLM counts the confusion matrix on which precision, recall, and F1 are computed.

There are two benchmark datasets:

| File | Description |
|---|---|
| `pubchef-test-fpbal-0.5.csv` | PubCheF held-out test set, reduced such that every molecule is at least 0.5 RDKit fingerprint Tanimoto different from one another |
| `opentargets-20240626.csv` | OpenTargets drug–indication dataset|

Both require columns: `id`, `smiles`, `ground_truth`.

Predictions must be in format: `id`, `smiles`, `prediction`.

**NOTE**: This benchmark eval requires an OPENAI_API_KEY and costs money to run. Running with gpt-5.4-mini costs ~$20 for three models on both benchmark datasets. We recommend looking into the current model offerings to decide which model to use and if you want to use a fine-tuned model or not. From our experiments, it seems that the results are fairly consistent across LLM judge models (confirmed for gpt-5.4-mini, which achieved 0.523 F1 compared to the 0.529 reported in the manuscript from using gpt-4-0613).


## Acknowledgements

The explainability module (`PubCheF-1/Transformer-Explainability/`) is adapted from [Transformer Interpretability Beyond Attention Visualization [CVPR 2021]](https://github.com/hila-chefer/Transformer-Explainability) by Hila Chefer et al. (MIT License).

---

## Citation

If you use PubCheF in your work, please cite:

```
@article{kosonocky2025pubchef,
  title={Learning from human and chemical languages to predict biological function},
  author={Kosonocky, Clayton W. and Kadeřábková, Nikol and Kim, Kangsan and Mahmood, Ayesha J.S. and Dunmyre, Alexander and Woolley, Phillip and Xing, Kristi and Winkler, Daniel and Babu, Tomer and Kadeřábek, Filip and Sessler, Jonathan L. and Anslyn, Eric V. and Marcotte, Edward M. and Zhang, Jessie and Ellington, Andrew D. and Mavridou, Despoina A.I.},
  journal={bioRxiv},
  year={2026},
  url={TBD}
}
```
