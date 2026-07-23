import os
import time
import random
import argparse
import pickle as pkl
from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

from joblib import Parallel, delayed

from model_utils import (
    SMILESTokenizer,
    load_model,
    load_checkpoint_weights,
    get_probs_from_model,
)
from utils import (
    canon_smiles,
    df_to_x_y,
)


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.environ['RDMAV_HUGEPAGES_SAFE'] = '1' # this is needed to avoid a warning from transformers


def calc_stats_from_probs(
                eval_predictions,
                eval_labels,
                n_labels,
                mlb,
                save_dir,
                split_type,
                split_name="test",
               ):
    """
    Get eval metrics given predictions. Track ROC-AUC and PR-AUC for each label and average across all labels.
    """
    roc_auc = []
    pr_auc = []
    brier_scores = []
    for i in range(n_labels):
        try:
            roc_auc.append(roc_auc_score(eval_labels[:, i], eval_predictions[:, i]))
            pr_auc.append(average_precision_score(eval_labels[:, i], eval_predictions[:, i]))
            brier_scores.append(brier_score_loss(eval_labels[:, i], eval_predictions[:, i]))

        except ValueError:
            roc_auc.append(0)
            pr_auc.append(0)
            brier_scores.append(brier_score_loss(eval_labels[:, i], eval_predictions[:, i]))

    print(f"Len of roc_auc: {len(roc_auc)}")
    print(f"Len of pr_auc: {len(pr_auc)}")
    print(f"Len of brier_scores: {len(brier_scores)}")
    print(f"Len of labels: {len(mlb.classes_)}")

    df = pd.DataFrame(
        {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier_scores,
            "labels": mlb.classes_,
        }
    )
    df.to_csv(save_dir / f"{split_type}_{split_name}_ensemble_metrics_indiv.csv", index=False)

    # ignore zero values as these are labels not found in the eval set
    mean_roc_auc = np.mean([roc for roc in roc_auc if roc != 0])
    mean_pr_auc = np.mean([pr for pr in pr_auc if pr != 0])
    mean_brier_score = np.mean(brier_scores)

    print(f"Mean ROC-AUC: {mean_roc_auc:.6f}")
    print(f"Mean PR-AUC:  {mean_pr_auc:.6f}")
    print(f"Mean Brier:   {mean_brier_score:.6f}")

    with open(save_dir / f"{split_type}_{split_name}_ensemble_metrics_mean.txt", "w") as f:
        f.write(f"Mean ROC-AUC: {mean_roc_auc:.6f}\n")
        f.write(f"Mean PR-AUC: {mean_pr_auc:.6f}\n")
        f.write(f"Mean Brier Score: {mean_brier_score:.6f}\n")


# NOTE This should be moved to its own script. Doesn't need to reside here.
def find_best_thresholds(y_true, y_probs):
    thresholds = np.arange(0.0, 1.01, 0.01)

    def process_label(label_idx):
        print(f"{label_idx} of {y_true.shape[1]}")
        best_f1 = 0.0
        best_threshold = 0.0

        for threshold in thresholds:
            y_pred = (y_probs[:, label_idx] >= threshold).astype(int)
            f1 = f1_score(y_true[:, label_idx], y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold, best_f1

    results = Parallel(n_jobs=-1)(delayed(process_label)(label_idx) for label_idx in range(y_true.shape[1]))

    best_thresholds, best_f1_scores = zip(*results)
    return np.array(best_thresholds), np.array(best_f1_scores)


def main(args):
    t_start = time.time()
    t_curr = t_start

    split_type = args.split_type
    data_dir = Path(args.data_dir)
    batch_size = args.batch_size
    base_model_name = args.base_model_name
    seed = args.seed
    random_canon = args.random_canon
    isomericSmiles = args.isomericSmiles
    kekuleSmiles = args.kekuleSmiles
    splits_to_eval = [s.strip() for s in args.splits.split(",")]

    save_dir = Path(f"models/{args.model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.model_name == "ensemble_single_canon_chiral_20epoch":
        run_names = ["exalted-sweep-1", "playful-sweep-2", "bumbling-sweep-3"]
    elif args.model_name == "ensemble_random_canon_chiral_20epoch":
        run_names = ["electric-sweep-1", "wandering-sweep-2", "denim-sweep-3"]
    else:
        run_names = [args.model_name]
    print(f"INFO: Evaluating model(s): {run_names}")

    if args.debug:
        val_df = pd.read_csv(data_dir / split_type / "val.csv", nrows=1000)
        test_df = pd.read_csv(data_dir / split_type / "test.csv", nrows=1000)
    else:
        val_df = pd.read_csv(data_dir / split_type / "val.csv")
        test_df = pd.read_csv(data_dir / split_type / "test.csv")

    val_df['labels'] = val_df['labels'].apply(literal_eval)
    test_df['labels'] = test_df['labels'].apply(literal_eval)

    if not ((isomericSmiles is True) and (kekuleSmiles is False)):
        print(f"INFO: Canonicalizing smiles strings. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
        val_df['smiles'] = val_df['smiles'].apply(lambda x: canon_smiles(x, isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles))
        val_df = val_df.dropna(subset=['smiles'])
        test_df['smiles'] = test_df['smiles'].apply(lambda x: canon_smiles(x, isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles))
        test_df = test_df.dropna(subset=['smiles'])

    with open(data_dir / "mlb.pkl", "rb") as f:
        mlb = pkl.load(f)

    X_val, y_val, cid_val = df_to_x_y(val_df, mlb)
    X_test, y_test, cid_test = df_to_x_y(test_df, mlb)
    print(f"INFO: Data loaded, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    tokenizer = SMILESTokenizer(vocab_path="tokenizer/vocab.json", download_vocab=True)

    model, device, d_model = load_model(
        drop_rate=0.1, d_out=len(mlb.classes_), base_model_name=base_model_name,
    )

    split_data = {
        "val":  (X_val,  y_val,  cid_val),
        "test": (X_test, y_test, cid_test),
    }

    for split_name in splits_to_eval:
        X, y, cid = split_data[split_name]
        print(f"\n=== Evaluating on {split_name} set ===")

        probs = []
        for run_name in run_names:
            load_checkpoint_weights(model, f"models/{run_name}/best_model.pt", device)
            print(f"INFO: Getting probs for {run_name} ({split_name}). Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
            split_predictions, split_labels = get_probs_from_model(
                model,
                X,
                tokenizer,
                device,
                t_curr,
                y=y,
                batch_size=batch_size,
                seed=seed,
                random_canon=random_canon,
            )
            probs.append(split_predictions)

        mean_probs = np.mean(probs, axis=0)

        cutoffs, f1_scores = find_best_thresholds(y, mean_probs)
        cutoff_df = pd.DataFrame({"labels": mlb.classes_, "cutoffs": cutoffs, "f1_scores": f1_scores})
        cutoff_df.to_csv(save_dir / f"{split_type}_{split_name}_ensemble_f1optimal_cutoffs.csv", index=False)

        calc_stats_from_probs(
            mean_probs,
            y,
            y.shape[1],
            mlb,
            save_dir,
            split_type,
            split_name,
        )

        predictions_df = pd.DataFrame(mean_probs, columns=mlb.classes_)
        predictions_df["smiles"] = X
        predictions_df["cid"] = cid
        predictions_df.to_csv(save_dir / f"{split_type}_{split_name}_ensemble_predictions.csv", index=False)
        print(f"INFO: Saved results for {split_name} set. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    print(f"\nINFO: Total time: {round(abs((t_old:=t_start) - (t_curr:=time.time())), 3)} seconds")


if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser(description="Evaluate PubCheF-1 ensemble on val/test splits")
    parser.add_argument("--model_name", default="ensemble_single_canon_chiral_20epoch",
                        help="'ensemble_single_canon_chiral_20epoch', 'ensemble_random_canon_chiral_20epoch', or a single run name in models/")
    parser.add_argument("--data_dir", default="../data/final_datasets/preprocessed_propagated_1_hard",
                        help="Directory containing split CSVs and mlb.pkl")
    parser.add_argument("--split_type", default="scaffold_split",
                        help="Subdirectory within data_dir containing train/val/test.csv")
    parser.add_argument("--splits", default="test",
                        help="Comma-separated list of splits to evaluate (e.g. 'val,test')")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_canon", action="store_true",
                        help="Apply random canonicalization during eval (data augmentation)")
    parser.add_argument("--base_model_name", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--isomericSmiles", default=True)
    parser.add_argument("--kekuleSmiles", default=False)
    parser.add_argument("--debug", action="store_true",
                        help="Load only first 1000 rows for quick testing")
    args = parser.parse_args()
    main(args)
