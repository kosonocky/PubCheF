#!/usr/bin/env python3
"""
Explainability Importance for SMILES string classification using ChemBERTa and LRP.
"""

import argparse
import os
import sys
import json
import random
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tokenizers.models import WordLevel
from tokenizers import Tokenizer, Regex
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import cm, colors
import matplotlib.pyplot as plt

# Try to properly path the custom Transformer explainability module
SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMER_EXP_DIR = SCRIPT_DIR / "Transformer-Explainability"
sys.path.insert(0, str(TRANSFORMER_EXP_DIR))

# Now import the customized BERT modules
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from BERT_explainability.modules.layers_ours import Linear, Dropout


class ChemMLMRegressor(nn.Module):
    """
    Classification on ChemBERTa
    """
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", drop_rate=0.2, d_out=1):
        super(ChemMLMRegressor, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.bert.config.hidden_size, d_out),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True
        )
        predictions = self.regressor(outputs["hidden_states"][-1][:, 0, :])  # state of cls token
        return {"predictions": predictions}


class SMILESTokenizer:
    """
    Tokenizer for SMILES strings. AutoTokenizer using DeepChem/ChemBERTa-77M-MLM was broken and this fixes it.
    Requires the vocab.json file from the DeepChem/ChemBERTa-77M-MLM model.
    """
    def __init__(self, vocab_path="tokenizer/vocab.json", download_vocab=True):
        if download_vocab:
            tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            Path("tokenizer").mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained("tokenizer")

        self.tokenizer = Tokenizer(WordLevel.from_file(vocab_path, unk_token='[UNK]'))
        self.tokenizer.pre_tokenizer = Split(pattern=Regex("Cl|Br|%[0-9]{2}|>>|\[(.*?)\]|."), behavior='isolated')
        self.tokenizer.add_special_tokens(["[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )

    def __call__(self, text, length_cutoff=510):
        encoded_corpus = [self.tokenizer.encode(t) for t in text]
        max_len = min(max((len(encoded.ids) for encoded in encoded_corpus)), length_cutoff)
        
        padded_corpus = np.zeros((len(encoded_corpus), max_len), dtype=np.int64)
        attention_masks = np.zeros_like(padded_corpus)

        for i, encoded in enumerate(encoded_corpus):
            padded_corpus[i, : len(encoded.ids[:max_len])] = encoded.ids[:max_len]
            attention_masks[i, : len(encoded.ids[:max_len])] = 1

        return {"input_ids": padded_corpus, "attention_mask": attention_masks}


def canon_smiles(smiles, isomericSmiles=True):
    try: 
        return Chem.CanonSmiles(smiles, useChiral=isomericSmiles)
    except Exception as e:
        print(f"Exception canonicalizing {smiles}: {e}")
        return smiles


def load_checkpoint_to_non_parallel(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    random.seed(seed)
    np.random.seed(seed)


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device: {device}")

    smiles = args.smiles
    name = args.name
    label_to_explain = args.label
    isomericSmiles = True
    model_name = "DeepChem/ChemBERTa-77M-MLM"

    c_smiles = canon_smiles(smiles, isomericSmiles=isomericSmiles)
    print("Entered SMILES is already canonicalized:", smiles == c_smiles)
    smiles = c_smiles

    # Setup directories relative to the script
    data_dir = SCRIPT_DIR.parent / "data" / "final_datasets" / "preprocessed_propagated_1_hard"
    models_dir = SCRIPT_DIR / "models"
    figs_dir = SCRIPT_DIR / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = TRANSFORMER_EXP_DIR / "element_tokens.json"

    # Define exact models used for the ensemble (extracted from original script)
    run_names = ["exalted-sweep-1", "playful-sweep-2", "bumbling-sweep-3"]

    tokenizer = SMILESTokenizer()
    
    with open(data_dir / "mlb.pkl", "rb") as f:
        mlb = pkl.load(f)

    # Initialize model using dims from ChemMLMRegressor
    roberta_model = ChemMLMRegressor(model_name=model_name, drop_rate=0.1, d_out=len(mlb.classes_))

    preds = []
    expls = []

    for run_name in run_names:
        model_path = models_dir / run_name / "best_model.pt"
        if not model_path.exists():
            print(f"WARNING: Model path {model_path} not found. Skipping {run_name}.")
            continue
            
        roberta_model = load_checkpoint_to_non_parallel(roberta_model, model_path)
        roberta_state_dict = roberta_model.state_dict()
        config = roberta_model.bert.config

        # Adapt the state dict from RoBERTa to match our customized BERT architecture
        new_state_dict = {}
        for key, value in roberta_state_dict.items():
            new_key = key.replace('bert.roberta', 'bert').replace('regressor.1', 'classifier')
            new_state_dict[new_key] = value
            if "lm_head" in new_key:
                new_state_dict.pop(new_key)

        new_state_dict["bert.embeddings.position_ids"] = torch.arange(2, 517).expand((1, -1)).to(device)
        new_state_dict["bert.pooler.dense.weight"] = torch.eye(384)
        new_state_dict["bert.pooler.dense.bias"] = torch.zeros(384)

        if label_to_explain == "base_embedding":
            new_state_dict["classifier.weight"][-1] = torch.ones(384)
            new_state_dict["classifier.bias"][-1] = torch.zeros(1)
            print("Using 'base embedding'")

        new_model = BertForSequenceClassification(config)
        new_model.classifier = Linear(config.hidden_size, len(mlb.classes_))
        new_model.dropout = Dropout(0)
        new_model.bert.embeddings.position_embeddings.padding_idx = 1
        new_model.load_state_dict(new_state_dict)
        new_model.to(device)
        new_model.eval()

        encoded_corpus = tokenizer([smiles])
        input_ids = torch.tensor(encoded_corpus["input_ids"]).to(device)
        attention_mask = torch.tensor(encoded_corpus["attention_mask"]).to(device)

        output = new_model(input_ids=input_ids, attention_mask=attention_mask)
        results_df = pd.DataFrame(output['logits'].detach().cpu().numpy(), columns=mlb.classes_).T
        preds.append(results_df)

        if label_to_explain == "base_embedding":
            target_class = -1
        else:
            if label_to_explain not in mlb.classes_.tolist():
                raise ValueError(f"Label '{label_to_explain}' not found in the classes.")
            target_class = mlb.classes_.tolist().index(label_to_explain)
            
        explanations = Generator(new_model)
        expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_class)[0]
        expls.append(expl.cpu().detach().numpy())

    if not preds:
        return print("No models were successfully loaded.")

    # Mean across runs
    results_df = pd.concat(preds, axis=1).mean(axis=1).to_frame()
    expls = np.mean(expls, axis=0)

    results_df = results_df.sort_values(by=0, ascending=False)
    print("\nTop Predictors:")
    for index, row in results_df.head(10).iterrows():
        print(f" {index} : {1 / (1 + np.exp(-row[0])):.4f}")

    with open(tokens_path, "r") as f:
        atom_tokens = json.load(f)

    # Get atom values
    atom_values = []
    for input_id, score in zip(input_ids[0].to('cpu'), expls):
        if int(input_id) in atom_tokens.values():
            atom_values.append(score.item())

    if not atom_values:
        print("Could not map tokens to atoms successfully.")
        return

    atom_values = np.array(atom_values)
    atom_val_range = atom_values.max() - atom_values.min()
    if atom_val_range == 0:
        atom_values = np.zeros_like(atom_values)
    else:
        atom_values = (atom_values - atom_values.min()) / atom_val_range

    # Create RDKit molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)

    cmap = plt.cm.Reds
    norm = colors.Normalize(vmin=0, vmax=max(atom_values))
    value_to_rgba = cm.ScalarMappable(cmap=cmap, norm=norm).to_rgba
    atom_colors = {i: value_to_rgba(value) for i, value in enumerate(atom_values)}

    # Drawing options
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.bondLineWidth = 1
    drawer.DrawMolecule(mol, highlightAtoms=list(range(len(atom_values))), highlightAtomColors=atom_colors, highlightBonds=[], highlightBondColors={})
    drawer.FinishDrawing()

    # Get SVG data and save
    svg = drawer.GetDrawingText().replace('svg:', '')
    safe_label = label_to_explain.replace('/', '_')
    svg_path = figs_dir / f"{name}_{safe_label}.svg"
    
    with open(svg_path, "w") as f:
        f.write(svg)
    print(f"\nSaved molecule SVG explanation to: {svg_path}")

    # Plot color bar
    fig, ax = plt.subplots(figsize=(7, 1.5))
    fig.subplots_adjust(bottom=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    plt.xticks(fontsize=20)
    target_name = "Base Embedding" if label_to_explain == "base_embedding" else mlb.classes_[target_class]
    cbar.set_label(f'"{target_name}" Importance', fontsize=20)
    plt.tight_layout()
    
    cbar_path = figs_dir / f"{name}_{safe_label}_colorbar.png"
    plt.savefig(cbar_path)
    print(f"Saved colorbar PNG to: {cbar_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainability Importance for SMILES using LRP")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string to explain")
    parser.add_argument("--name", type=str, default="molecule", help="ID or name for saving the figure")
    parser.add_argument("--label", type=str, required=True, help="Label to explain (e.g. 'Beta-Lactamase Inhibitor')")
    args = parser.parse_args()
    main(args)
