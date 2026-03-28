import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rdkit import Chem


def canon_smiles(smiles, isomericSmiles=True, kekuleSmiles=False):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles)
    except Exception as e:
        print(e)
        print(f"ERROR: {smiles} set to None")
        return None

def canon_smiles_single_random(smiles, isomericSmiles=True):
    """
    Canonicalize a SMILES string by choosing a random atom as the root. Works for each part of a multi-molecule SMILES string complex
    """
    try:
        final_str = ""
        for s in smiles.split("."):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                final_str += s
                final_str += "."
                continue
            atom_count = sum([1 for atom in mol.GetAtoms()])
            random_choice = random.choice(range(atom_count))
            final_str += Chem.MolToSmiles(mol, canonical=True, rootedAtAtom=random_choice, isomericSmiles=isomericSmiles)
            final_str += "."
        final_str = final_str[:-1]
        return final_str
    except Exception as e:
        print(f"Triggered exception: {e} for smiles: {smiles}. Returning original smiles.")
        return smiles

def shuffle_tokens(string, tokenizer):
    """
    Shuffle string by tokenizing, shuffling, and decoding back into string
    """
    ids = tokenizer.tokenizer.encode(string).ids
    random.shuffle(ids)
    return "".join(tokenizer.tokenizer.decode(ids).split())

def smooth_labels(y, alpha=0.1):
    """
    Label smoothing
    """
    with torch.no_grad():
        y = y * (1 - alpha) + (alpha / y.shape[1])
    return y

def df_to_x_y(df, mlb):
    y = mlb.transform(df['labels'])
    print(f"INFO: y.shape: {y.shape}")
    X = df['smiles'].values
    cid = df["cid"].values
    return X, y, cid

def save_plot_losses(train_losses, validation_losses, save_path):
    np.save(save_path / "train_losses.npy", np.array(train_losses))
    np.save(save_path / "validation_losses.npy", np.array(validation_losses))
    pd.DataFrame(
        {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
    ).to_csv(save_path / "train_val_losses.csv", index=False)

    # plot losses with log scale y axis
    plt.plot(train_losses, label="Train loss")
    plt.plot(validation_losses, label="Validation loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path / "losses.png")
    plt.close()
