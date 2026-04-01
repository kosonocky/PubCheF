"""
Post-process the output of gpt_eval.py: merge per-chunk CSVs and compute
precision, recall, and F1.

Expected input:
  One or more eval output directories from gpt_eval.py, each containing:
    - samples_0.csv, samples_1000.csv, ... (chunk output files)
  Each CSV has columns: ground_truth, prediction, chat_response, TP, FP, FN

Usage examples:

  # Score a single eval run:
  python process_eval_results.py \\
      --eval "chef-v1:chef/chef-v1/chef_eval_gpt-4-turbo-2024-04-09" \\
      --output results/scores.txt

  # Compare multiple models:
  python process_eval_results.py \\
      --eval "chef-v1:chef/chef-v1/chef_eval_gpt-4-turbo-2024-04-09" \\
      --eval "chef-v2:chef/chef-v2/chef_eval_gpt-4-turbo-2024-04-09" \\
      --eval "molt5:molt5/chef_eval_gpt-4-turbo-2024-04-09" \\
      --output results/scores.txt
"""

import argparse
from pathlib import Path

import pandas as pd


def import_eval_dir(eval_dir: Path) -> pd.DataFrame:
    """Merge all samples_*.csv files in eval_dir, sorted by chunk index."""
    chunk_files = [f for f in eval_dir.iterdir() if f.name.startswith("samples_") and f.suffix == ".csv"]
    if not chunk_files:
        raise FileNotFoundError(f"No samples_*.csv files found in {eval_dir}")
    chunk_files.sort(key=lambda f: int(f.stem.split("_")[1]))
    df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    print(f"  [{eval_dir.name}] Loaded {len(df)} rows from {len(chunk_files)} chunk(s)")
    return df


def compute_scores(df: pd.DataFrame) -> dict:
    """Compute micro-averaged precision, recall, and F1 from TP/FP/FN columns."""
    for col in ["TP", "FP", "FN"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_failed = df[["TP", "FP", "FN"]].isna().any(axis=1).sum()
    if n_failed:
        print(f"    Skipping {n_failed} rows with missing TP/FP/FN (failed GPT calls)")
    df = df.dropna(subset=["TP", "FP", "FN"])

    tp = df["TP"].sum()
    fp = df["FP"].sum()
    fn = df["FN"].sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_samples": len(df),
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_results(all_scores: dict) -> str:
    col_w = max(len(k) for k in all_scores) + 2
    header = f"{'Model':<{col_w}}  {'N':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>8}  {'FP':>8}  {'FN':>8}"
    sep = "-" * len(header)
    rows = [header, sep]
    for name, s in all_scores.items():
        rows.append(
            f"{name:<{col_w}}  {s['n_samples']:>6}  "
            f"{s['precision']:>10.4f}  {s['recall']:>8.4f}  {s['f1']:>8.4f}  "
            f"{s['TP']:>8}  {s['FP']:>8}  {s['FN']:>8}"
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval", action="append", dest="evals", metavar="NAME:PATH", required=True,
        help=(
            "Eval run to process, as 'name:path'. Path should be a directory "
            "containing samples_*.csv files from gpt_eval.py. "
            "Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--output", default="results.txt",
        help="Path to write the final scores (default: results.txt).",
    )
    args = parser.parse_args()

    eval_entries = {}
    for entry in args.evals:
        if ":" not in entry:
            parser.error(f"--eval must be NAME:PATH format, got: {entry!r}")
        name, _, path = entry.partition(":")
        eval_entries[name] = Path(path)

    print("=== Importing chunk CSVs ===")
    all_scores = {}
    for name, path in eval_entries.items():
        df = import_eval_dir(path)
        scores = compute_scores(df)
        all_scores[name] = scores
        print(f"    P={scores['precision']:.4f}  R={scores['recall']:.4f}  F1={scores['f1']:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_text = format_results(all_scores)
    output_path.write_text(result_text + "\n")

    print(f"\nResults written to {output_path}\n")
    print(result_text)


if __name__ == "__main__":
    main()
