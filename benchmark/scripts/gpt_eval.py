"""
GPT-based evaluation of molecule attribute predictions.

Input files use standardised column names:
  predictions CSV : id, smiles, prediction
  dataset CSV     : id, smiles, ground_truth

Pass --dataset to join ground truth from the dataset file on 'id'.
If your predictions file already contains a ground_truth column, omit --dataset.

Example usage:
  # Label-based model (e.g. ChEF), ground truth in separate dataset file:
  python gpt_eval.py \\
      --input     predictions/pubchef-test-fpbal-0.5_chef-fplr.csv \\
      --dataset   dataset/pubchef-test-fpbal-0.5.csv \\
      --model-type labels \\
      --output-dir results/chef-fplr_pubchef

  # Free-text captioning model (e.g. MolT5), same setup:
  python gpt_eval.py \\
      --input     predictions/pubchef-test-fpbal-0.5_molt5.csv \\
      --dataset   dataset/pubchef-test-fpbal-0.5.csv \\
      --model-type free-text \\
      --output-dir results/molt5_pubchef
"""

import argparse
import asyncio
import time
from functools import partial
from pathlib import Path

import openai
import pandas as pd
import backoff

client = openai.Client()

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

PROMPTS = {
    "labels": {
        "system": (
            "You are a skilled medicinal chemist evaluating a model's predictions "
            "of a molecule against the ground-truth"
        ),
        "user": (
            r"The following are predicted attributes of about the said molecule:"
            r"\n'{__predictions__}',"
            r"\nThe following are the true attributes of this molecule:"
            r"\n'{__actual__}"
            r"\nIgnore all claims about the molecule's structure and only focus on its "
            r"functional attributes. Compute the TP/FP/FN of the predicted attributes "
            r"given the true attributes. Finish your response with the exact form:"
            r"\n'TP:a,FP:b,FN:c'"
        ),
    },
    "free-text": {
        "system": (
            "You are a skilled medicinal chemist evaluating a model's predictions "
            "of a molecule against the ground-truth"
        ),
        "user": (
            r"Break this prediction down into a list of the individual claims about the said molecule:"
            r"\n'{__predictions__}',"
            r"\nThe following are the true attributes of this molecule:"
            r"\n'{__actual__}"
            r"\nIgnore all claims about the molecule's structure and only focus on its "
            r"functional attributes. Compute the TP/FP/FN of the predicted claims "
            r"given the true attributes. Finish your response with the exact form:"
            r"\n'TP:a,FP:b,FN:c'"
        ),
    },
}

# ------------------------------------------------------------------
# OpenAI helpers
# ------------------------------------------------------------------

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def completions_with_backoff(**kwargs):
    loop = asyncio.get_running_loop()
    func = partial(client.chat.completions.create, **kwargs)
    result = await loop.run_in_executor(None, func)
    return result


async def get_chat_response(
    system_message: str,
    user_request: str,
    model: str,
    seed: int,
    temperature: float,
):
    try:
        if not system_message or not user_request or not model:
            print("ERROR: MISSING PROMPT OR MODEL")
            return None
        response = await completions_with_backoff(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_request},
            ],
            seed=seed,
            stream=False,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# ------------------------------------------------------------------
# Processing
# ------------------------------------------------------------------

async def process_single_sample(
    ground_truth: str,
    prediction: str,
    gpt_model: str,
    system_message: str,
    user_request_partial: str,
    seed: int,
    temperature: float,
):
    user_request = (
        user_request_partial
        .replace("__predictions__", prediction)
        .replace("__actual__", ground_truth)
    )
    chat_response = await get_chat_response(
        system_message=system_message,
        user_request=user_request,
        model=gpt_model,
        seed=seed,
        temperature=temperature,
    )
    return {"ground_truth": ground_truth, "prediction": prediction, "chat_response": chat_response}


async def process_samples(
    ground_truth_subset,
    predictions_subset,
    gpt_model: str,
    system_message: str,
    user_request_partial: str,
    seed: int,
    temperature: float,
):
    tasks = [
        process_single_sample(gt, pred, gpt_model, system_message, user_request_partial, seed, temperature)
        for gt, pred in zip(ground_truth_subset, predictions_subset)
    ]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


def extract_tpfpfn(results_df: pd.DataFrame) -> pd.DataFrame:
    """Extract TP/FP/FN from the chat_response column, trying with and without spaces."""
    for pattern in [r"TP:(\d+),FP:(\d+),FN:(\d+)", r"TP:(\d+), FP:(\d+), FN:(\d+)"]:
        extracted = results_df["chat_response"].str.extract(pattern, expand=True)
        for col, idx in [("TP", 0), ("FP", 1), ("FN", 2)]:
            if col not in results_df.columns:
                results_df[col] = extracted[idx]
            else:
                results_df[col] = results_df[col].fillna(extracted[idx])
    return results_df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Predictions CSV with columns: id, smiles, prediction.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help=(
            "Dataset CSV with columns: id, smiles, ground_truth. "
            "When provided, joined to --input on 'id' to supply ground truth."
        ),
    )
    parser.add_argument(
        "--model-type", required=True, choices=["labels", "free-text"],
        help=(
            "'labels' for structured/multi-label predictions (e.g. ChEF); "
            "'free-text' for captioning models (e.g. MolT5)."
        ),
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where per-chunk result CSVs and the prompt will be saved.",
    )
    parser.add_argument(
        "--gpt-model", default="gpt-4-turbo-2024-04-09",
        help="OpenAI model to use for evaluation (default: gpt-4-turbo-2024-04-09).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Number of samples to process per async batch (default: 1000).",
    )
    args = parser.parse_args()

    t_start = time.time()
    t_curr = t_start

    print(f"INFO: model-type   = {args.model_type}")
    print(f"INFO: gpt-model    = {args.gpt_model}")
    print(f"INFO: temperature  = {args.temperature}")
    print(f"INFO: seed         = {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    if args.dataset:
        dataset_df = pd.read_csv(args.dataset)[["id", "ground_truth"]]
        df = df.merge(dataset_df, on="id", how="inner")

    missing = [c for c in ["ground_truth", "prediction"] if c not in df.columns]
    if missing:
        parser.error(
            f"Column(s) not found after loading inputs: {missing}. "
            f"Available columns: {list(df.columns)}. "
            f"Use --dataset to join ground truth from a separate file."
        )

    df = df.dropna(subset=["ground_truth", "prediction"]).reset_index(drop=True)
    print(f"INFO: {len(df)} samples after dropping NaNs")

    ground_truth = df["ground_truth"].astype(str)
    predictions = df["prediction"].astype(str)

    system_message = PROMPTS[args.model_type]["system"]
    user_request_partial = PROMPTS[args.model_type]["user"]

    with open(output_dir / "prompt.txt", "w") as f:
        f.write(system_message + "\n\n" + user_request_partial)

    for i in range(0, len(ground_truth), args.chunk_size):
        lim = min(args.chunk_size, len(ground_truth) - i)
        elapsed = round(abs((t_curr := time.time()) - t_curr), 3)
        print(f"Processing rows {i} to {i + lim}... Time: {elapsed}s")

        gt_chunk = ground_truth.iloc[i : i + lim]
        pred_chunk = predictions.iloc[i : i + lim]

        results_df = await process_samples(
            gt_chunk, pred_chunk,
            args.gpt_model, system_message, user_request_partial,
            args.seed, args.temperature,
        )
        results_df = extract_tpfpfn(results_df)
        results_df.to_csv(output_dir / f"samples_{i}.csv", index=False)

    print("Complete.")
    print(f"Total time elapsed: {round(time.time() - t_start, 1)}s")


if __name__ == "__main__":
    print(__file__)
    asyncio.run(main())
