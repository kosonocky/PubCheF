import os
import time
import random
import pickle as pkl
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_utils import (
    SMILESTokenizer,
    load_model,
    load_checkpoint_weights,
    get_probs_from_model,
)
from utils import canon_smiles


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# keeping off because it slows down training
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

os.environ['RDMAV_HUGEPAGES_SAFE'] = '1' # this is needed to avoid a warning from transformers


def main(args):
    t_start = time.time()
    t_curr = t_start
    isomericSmiles = args.isomericSmiles
    mlb_dir = Path(args.mlb_dir)
    model_name = Path(f"models/{args.model_name}")

    # unpack args
    if args.input_csv:
        input_name = Path(args.input_csv).stem
        input_df = pd.read_csv(args.input_csv)
        if args.smiles_column not in input_df.columns:
            raise ValueError(f"SMILES column '{args.smiles_column}' not found in CSV.")

        input_df["canon_smiles"] = input_df[args.smiles_column].apply(lambda x: canon_smiles(x, isomericSmiles=isomericSmiles))
        input_df = input_df.dropna(subset=["canon_smiles"]).reset_index(drop=True)
        X_smiles = input_df["canon_smiles"].tolist()
    else:
        input_name = "single_smiles"
        X_smiles = [canon_smiles(args.smiles, isomericSmiles=isomericSmiles)]

        none_indices = [i for i, smiles in enumerate(X_smiles) if smiles is None]
        X_smiles = [smiles for i, smiles in enumerate(X_smiles) if i not in none_indices]

    print(f"\n===Input Params===")
    print(f"INFO: Input target: {args.input_csv or args.smiles}")
    print(f"INFO: p_threshold: {args.p_threshold}")
    print(f"INFO: batch_size_chunk: {args.batch_size}")
    print(f"INFO: extract_embeddings: {args.extract_embeddings}")
    print(f"\n===Training Params===")
    print(f"INFO: batch_size: {(batch_size:=256)}")
    print(f"INFO: gpus available: {torch.cuda.device_count()}")
    print(f"INFO: base_model_name: {(base_model_name:='DeepChem/ChemBERTa-77M-MLM')}")
    print(f"INFO: model_name: {model_name}")

    seed = 42

    # load mlb
    with open(mlb_dir / "mlb.pkl", "rb") as f:
        mlb = pkl.load(f)

    # tokenize data, get masks, and create dataloaders
    tokenizer = SMILESTokenizer(vocab_path="tokenizer/vocab.json", download_vocab=True)
    model, device, d_model = load_model(
        drop_rate=0.1, d_out=mlb.classes_.shape[0], base_model_name=base_model_name,
    )

    if model_name == Path("models/ensemble_single_canon_chiral_20epoch"):
        run_names = ["exalted-sweep-1", "playful-sweep-2", "bumbling-sweep-3"]
    else:
        print(f"INFO: Running inference for the single model: {model_name}")
        run_names = [model_name.name]

    if args.extract_embeddings:
        # Take the first model from the ensemble for extraction, or use a specific single model
        run_names = [run_names[0]]

    # Process in batches
    chunk_size = args.batch_size
    for i in range(0, len(X_smiles), chunk_size):
        X_smiles_batch = X_smiles[i:i+chunk_size]
        batch_end = min(i+chunk_size, len(X_smiles))

        if args.input_csv:
            df_batch = input_df.iloc[i:batch_end].copy()
            df_batch = df_batch.drop(columns=["canon_smiles"])
        else:
            df_batch = pd.DataFrame()
            df_batch["smiles"] = X_smiles_batch
            df_batch["compound_id"] = args.compound_id or "molecule_0"

        print(f"INFO: Getting predictions for batch {i} to {batch_end}. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

        preds = []
        for run_name in run_names:
            # load best model
            model_path = model_name if model_name.name == run_name else Path(f"models/{run_name}")
            load_checkpoint_weights(model, model_path / "best_model.pt", device)
            print(f"INFO: Getting probs from model {run_name}. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
            predictions = get_probs_from_model(
                model,
                X_smiles_batch,
                tokenizer,
                device,
                t_curr,
                batch_size=batch_size,
                seed=seed,
                extract_embeddings=args.extract_embeddings,
            )

            if args.extract_embeddings:
                preds.append(predictions) # For embeddings we just use first model
                break
            else:
                preds.append(pd.DataFrame(predictions, columns=mlb.classes_))

        if args.extract_embeddings:
            embeddings_arr = preds[0]

            # Save metadata (the batch dataframe) and embeddings together in a .pt dictionary
            batch_data = {
                "metadata": df_batch,
                "embeddings": torch.tensor(embeddings_arr)
            }

            if args.input_csv:
                save_dir = Path("inference_results", model_name.name, "embeddings", input_name)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(batch_data, save_dir / f"{input_name}_{i}_{batch_end}_embeddings.pt")
                print(f"Saved embeddings and metadata to: {save_dir / f'{input_name}_{i}_{batch_end}_embeddings.pt'}")
            else:
                save_dir = Path("inference_results", model_name.name, "embeddings", "single_smiles")
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(batch_data, save_dir / f"single_smiles_{i}_{batch_end}_embeddings.pt")
                print(f"Saved embeddings and metadata to: {save_dir / f'single_smiles_{i}_{batch_end}_embeddings.pt'}")
        else:
            # average predictions across models
            mean_probs = np.mean(preds, axis=0)

            if args.p_threshold is not None:
                # Filter significantly terms
                high_prob_indices = mean_probs >= args.p_threshold
                filtered_dict_list = []
                for j in range(len(high_prob_indices)):
                    filtered_probs = mean_probs[j][high_prob_indices[j]]
                    filtered_labels = mlb.classes_[high_prob_indices[j]]
                    filtered_dict = {
                        filtered_labels[k]: float(f"{float(filtered_probs[k]):.3g}")
                        for k in range(len(filtered_probs))
                    }
                    filtered_dict_list.append(filtered_dict)
                df_batch["top_preds"] = filtered_dict_list
            else:
                # include all probs like originally
                preds_df = pd.DataFrame(mean_probs, columns=mlb.classes_)
                df_batch = pd.concat([df_batch, preds_df], axis=1)

            if args.input_csv:
                save_dir = Path("inference_results", model_name.name, "predictions", input_name)
                save_dir.mkdir(parents=True, exist_ok=True)
                df_batch.to_csv(save_dir / f"{input_name}_{i}_{batch_end}.csv", index=False)
            else:
                save_dir = Path("inference_results", model_name.name, "predictions", "single_smiles")
                save_dir.mkdir(parents=True, exist_ok=True)
                df_batch.to_csv(save_dir / f"single_smiles_{i}_{batch_end}.csv", index=False)

    print(f"INFO: Total time: {round(abs((t_old:=t_start) - (t_curr:=time.time())), 3)} seconds")


if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", type=str, help="Single SMILES string for prediction")
    input_group.add_argument("--input_csv", type=str, help="Path to a CSV file containing SMILES strings")

    parser.add_argument("--compound_id", type=str, default="", help="Literal ID or name for the single molecule (if using --smiles)")
    parser.add_argument("--smiles_column", type=str, default="smiles", help="Name of the SMILES column (if using --input_csv)")

    parser.add_argument("--isomericSmiles", default=True, help="chiral or achiral for smiles string canonicalization")
    parser.add_argument("--mlb_dir", default="../../../data/final_datasets/preprocessed_propagated_1_hard", help="directory of mlb. This is typically where the dataset is stored")
    parser.add_argument("--model_name", default="ensemble_single_canon_chiral_20epoch", help="Must be either 'ensemble_single_canon_chiral_20epoch' or a model directory containing a trained model within the models directory.")
    parser.add_argument("--p_threshold", type=float, default=None, help="probability threshold for saving only significant terms. If set, predictions below this are discarded.")
    parser.add_argument("--batch_size", type=int, default=100000, help="batch size for processing huge files")
    parser.add_argument("--extract_embeddings", action="store_true", help="whether to extract embeddings instead of making predictions")
    args = parser.parse_args()
    main(args)
