import sys
import json
import random
import argparse
from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Add parent directory to path to import shared model/utils modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import ChemMLMRegressor, SMILESTokenizer
from utils import canon_smiles

from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from BERT_explainability.modules.layers_ours import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_checkpoint_to_non_parallel(model, checkpoint_path):
    # Load the state dict saved with nn.DataParallel
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Create a new state dict without the 'module.' prefix
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load the modified state dict into the model
    model.load_state_dict(new_state_dict)

    return model


def main(args):
    smiles = args.smiles
    name = args.name
    label_to_explain = args.label_to_explain
    seed = args.seed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_name == "ensemble_single_canon_chiral_20epoch":
        run_names = ["exalted-sweep-1", "playful-sweep-2", "bumbling-sweep-3"]
        print(f"INFO: Running inference for the ensemble of models: {run_names}")
    else:
        run_names = [args.model_name]
        print(f"INFO: Running inference for the single model: {args.model_name}")
    save_dir = Path(f"../figs/{args.model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    smiles = canon_smiles(smiles)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"INFO: base_model_name: {(base_model_name:='DeepChem/ChemBERTa-77M-MLM')}")

    tokenizer = SMILESTokenizer(vocab_path=args.tokenizer_path)
    with open(Path(args.mlb_dir) / "mlb.pkl", "rb") as f:
        mlb = pkl.load(f)

    # initialize ChemMLMRegressor to load checkpoint weights and extract config/state_dict
    roberta_model = ChemMLMRegressor(
        base_model_name=base_model_name,
        drop_rate=0.1,
        d_out=len(mlb.classes_)
    )

    preds = []
    expls = []

    for run_name in run_names:
        model_dir = Path(f"../models/{run_name}")
        # load checkpoint but dont adapt
        roberta_model = load_checkpoint_to_non_parallel(roberta_model, model_dir / "best_model.pt")
        roberta_state_dict = roberta_model.state_dict()
        config = roberta_model.bert.config

        # NOTE Adapt the state dict from RoBERTa to work for BERT
        new_state_dict = {}
        for key, value in roberta_state_dict.items():
            new_key = key.replace('bert.roberta', 'bert')
            new_key = new_key.replace('regressor.1', 'classifier')

            new_state_dict[new_key] = value

            # remove lm_head
            if "lm_head" in new_key:
                new_state_dict.pop(new_key)

        # add position ids. Make it start at 2 and go to 517 to match roberta
        new_state_dict["bert.embeddings.position_ids"] = torch.arange(2, 517).expand((1, -1)).to(device)

        # create a custom pooler weight and bias that corresponds to just the cls token
        new_state_dict["bert.pooler.dense.weight"] = torch.eye(384).to(device) # this is fine because the pooler is just a linear layer. The pooler is also not used in the forward pass
        new_state_dict["bert.pooler.dense.bias"] = torch.zeros(384).to(device) # this is fine because the pooler is just a linear layer. The pooler is also not used in the forward pass

        if label_to_explain == "base_embedding":
            new_state_dict["classifier.weight"][-1] = torch.ones(384)
            new_state_dict["classifier.bias"][-1] = torch.zeros(1)
            print("Using 'base embedding'")

        # load from new_state_dict and config
        new_model = BertForSequenceClassification(config)

        # change output features of classifier
        new_model.classifier = Linear(config.hidden_size, len(mlb.classes_))

        # change dropout layer to 0
        new_model.dropout = Dropout(0)

        # set position embeddings padding_idx to 1
        new_model.bert.embeddings.position_embeddings.padding_idx = 1

        # load state dict
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
            target_class = mlb.classes_.tolist().index(label_to_explain)
        explanations = Generator(new_model)

        # options for generating explanations. In the paper, we used "generate_LRP".
        expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_class)[0]
        # expl = explanations.generate_LRP_last_layer(input_ids=input_ids, attention_mask=attention_mask, index=target_class)[0]
        # expl = explanations.generate_rollout(input_ids=input_ids, attention_mask=attention_mask, index=target_class)[0]
        # expl = explanations.generate_full_lrp(input_ids=input_ids, attention_mask=attention_mask, index=target_class)[0]
        # expl = explanations.generate_attn_gradcam(input_ids=input_ids, attention_mask=attention_mask, index=target_class)[0]
        expls.append(expl.cpu().detach().numpy())

    # mean across runs
    results_df = pd.concat(preds, axis=1).mean(axis=1).to_frame()
    expls = np.mean(expls, axis=0)

    results_df = results_df.sort_values(by=0, ascending=False)
    for index, row in results_df.iloc[:50].iterrows():
        print(f"{index} : {1 / (1 + np.exp(-row[0]))}") # print index and sigmoid of score

    with open("element_tokens.json", "r") as f:
        atom_tokens = json.load(f)

    rev_atom_tokens = {v: k for k, v in atom_tokens.items()}

    # get atom values
    atom_values = []
    for input_id, score in zip(input_ids[0].to('cpu'), expls):
        if int(input_id) in atom_tokens.values():
            atom_values += [score.item()]

    atom_values = np.array(atom_values)
    atom_values = (atom_values - atom_values.min()) / (atom_values.max() - atom_values.min()) # normalize scores

    # Create RDKit molecule from SMILES
    mol = Chem.MolFromSmiles(smiles) # smiles should be canonicalized already

    # create oneslope norm that goes from 0 to max. Make the max not so dark
    cmap = plt.cm.Reds
    norm = colors.Normalize(vmin=0, vmax=max(atom_values))
    # norm = colors.Normalize(vmin=0, vmax=1)

    # Create a function to convert values to colors
    value_to_rgba = cm.ScalarMappable(cmap=cmap, norm=norm).to_rgba

    # Apply this function to generate atom colors
    atom_colors = {i: value_to_rgba(value) for i, value in enumerate(atom_values)}

    # Drawing options
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()

    # Draw the molecule
    opts.bondLineWidth = 1
    drawer.DrawMolecule(mol, highlightAtoms=range(len(atom_values)), highlightAtomColors=atom_colors, highlightBonds=[], highlightBondColors={})
    drawer.FinishDrawing()

    # Get the SVG data and display the molecule
    svg = drawer.GetDrawingText().replace('svg:', '')

    # save fig
    Path("../figs").mkdir(parents=True, exist_ok=True)
    with open(f"../figs/{name}_{label_to_explain.replace('/','_')}.svg", "w") as f:
        f.write(svg)
    print(f"Saved molecule SVG to: ../figs/{name}_{label_to_explain.replace('/','_')}.svg")

    # Create and display the color bar
    fig, ax = plt.subplots(figsize=(7, 1.5))
    fig.subplots_adjust(bottom=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal', label=f"LRP Importance for {mlb.classes_[target_class]}")
    plt.xticks(fontsize=20)
    cbar.set_label(f'"{mlb.classes_[target_class]}" Importance', fontsize=20)

    plt.tight_layout()
    colorbar_path = f"../figs/{label_to_explain.replace('/','_').replace(' ', '_')}_colorbar.png"
    plt.savefig(colorbar_path)
    print(f"Saved colorbar to: {colorbar_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LRP-based token importance for PubCheF-1 predictions")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the molecule to explain")
    parser.add_argument("--name", type=str, required=True, help="Name of the molecule (used in output filenames)")
    parser.add_argument("--label_to_explain", type=str, required=True, help="Label to explain. Must be one of the labels in the dataset or 'base_embedding' to use the base embedding as the target.")
    parser.add_argument("--model_name", default="ensemble_single_canon_chiral_20epoch", help="Must be either 'ensemble_single_canon_chiral_20epoch' or a run name within the models directory.")
    parser.add_argument("--mlb_dir", default="../../data/final_datasets/preprocessed_propagated_1_hard", help="Directory containing mlb.pkl")
    parser.add_argument("--tokenizer_path", default="../tokenizer/vocab.json", help="Path to tokenizer vocab.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
