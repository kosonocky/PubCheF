import time
import pandas as pd
from pathlib import Path
import asyncio
import openai
import backoff
from collections import Counter
import argparse

def remove_special_characters(text:str):
    """
    Simplify text by removing special characters and lowercasing
    """
    text = text.replace("α", "Alpha").replace("β", "Beta").replace("γ", "Gamma").replace("δ", "Delta").replace("ε", "Epsilon").replace("ζ", "Zeta").replace("η", "Eta").replace("θ", "Theta").replace("ι", "Iota").replace("κ", "Kappa").replace("λ", "Lambda").replace("μ", "Mu").replace("ν", "Nu").replace("ξ", "Xi").replace("ο", "Omicron").replace("π", "Pi").replace("ρ", "Rho").replace("σ", "Sigma").replace("τ", "Tau").replace("υ", "Upsilon").replace("φ", "Phi").replace("χ", "Chi").replace("ψ", "Psi").replace("ω", "Omega")
    text = text.replace("₁", "1").replace("₂", "2").replace("₃", "3").replace("₄", "4").replace("₅", "5").replace("₆", "6").replace("₇", "7").replace("₈", "8").replace("₉", "9").replace("₀", "0")
    text = text.replace("¹", "1").replace("²", "2").replace("³", "3").replace("⁴", "4").replace("⁵", "5").replace("⁶", "6").replace("⁷", "7").replace("⁸", "8").replace("⁹", "9").replace("⁰", "0")
    return text

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def embed_with_backoff(client, **kwargs):
    return client.embeddings.create(**kwargs)

async def embed_text(client, text:str, model:str = "text-embedding-3-large"):
    """
    Embed text using OpenAI API
    """
    try:
        response = embed_with_backoff(
            client,
            input=text,
            model=model,
        )
        return response.data[0].embedding
    except Exception as e:
        print(e)
        return None


async def async_embed_text(client, text_list:list, model:str = "text-embedding-3-large"):
    """
    Embed a list of text using OpenAI API. This is done in parallel using asyncio. It creates a task for each text, awaits the results, gathers all at once, and returns a dataframe of text and embeddings.
    NOTE: Currently embedding lowercased text
    """
    tasks = [embed_text(client, text, model) for text in text_list]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame({"text": text_list, "embedding": results})


def merge_vocab_by_capitalization(vocab:list):
    vocab = list(vocab)
    # sort by capitalization (low to high). This ensures that the most capitalized version of a term is used
    vocab.sort(key=lambda x: x.lower())

    # group regardless of capitalization
    vocab_map = dict()
    for term in vocab:
        vocab_map.update({term: term.lower()})
    
    vocab = [vocab_map[term] for term in vocab]
    vocab = set(vocab) # remove duplicates

    vocab_reverse_map = dict()
    for k, v in vocab_map.items():
        if v not in vocab_reverse_map:
            # make the k the most capitalized version of v
            vocab_reverse_map.update({v: k})

    # map back to original capitalization
    vocab = [vocab_reverse_map[term] for term in vocab]

    vocab = set(vocab) # remove duplicates

    return list(vocab), vocab_reverse_map


async def main(args):
    t_start = time.time()
    t_curr = t_start
    client = openai.OpenAI()  # reads OPENAI_API_KEY from environment
    embedding_model = args.embedding_model
    # Load dataset of pmids and their summaries
    # This csv should contain the columns "pmid" and "chat_response"
    dataset_dir = Path(f"../../data/dataset_creation/bioassay_pmids/{args.gpt_model}")
    dataset_fname = "pmid_func_complete.csv"
    assert Path(dataset_dir, dataset_fname).exists(), f"ERROR: Dataset or dir does not exist: {Path(dataset_dir, dataset_fname)}"
    df = pd.read_csv(Path(dataset_dir, dataset_fname))
    print(f"INFO: Dataset loaded from {Path(dataset_dir, dataset_fname)}")

    df = df[["pmid", "chat_response"]]
    df = df.rename(columns={"chat_response": "labels"})
    df = df.dropna(subset=["labels"])
    print(f"INFO: {len(df)} rows after dropping NaN labels")
    df["labels"] = df["labels"].apply(lambda x: x.split(" / ")) # split into list of strings
    df["labels"] = df["labels"].apply(lambda x: [y for y in x if y != ""]) # remove empty strings
    df["labels"] = df["labels"].apply(lambda x: list(set([remove_special_characters(y) for y in x])))
    print(f"INFO: Saving formatted dataset (split into list of strings)")

    # create a set of each word in words
    vocab = set()
    for terms in df["labels"]:
        for term in terms:
            vocab.add(term)
    print(f"INFO: There are {len(vocab)} unique words in dataset. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    
    vocab, capitalization_mapping = merge_vocab_by_capitalization(vocab)

    # map if key in capitalization_mapping, if not keep original
    df["labels"] = df["labels"].apply(lambda x: list(set([capitalization_mapping[y] if y in capitalization_mapping else y for y in x])))
    df.to_csv(Path(dataset_dir, f"{dataset_fname.split('.')[0]}_formatted.csv"), index=False)

    print(f"INFO: Vocab size after merging by capitalization: {len(vocab)}")

    # save vocab to txt file
    with open(Path(dataset_dir, f"{dataset_fname.split('.')[0]}_vocab.txt"), "w") as f:
        f.write("\n".join(vocab))
    
    print(f"INFO: Saved vocab to txt file. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # embed vocab
    print(f"INFO: Embedding {len(vocab)} terms... Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    df = await async_embed_text(client, vocab, model=embedding_model)
    print(f"INFO: Finished embedding {len(vocab)} terms. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # save vocab embeddings to csv
    df.to_csv(Path(dataset_dir, f"{dataset_fname.split('.')[0]}_vocab_embeddings.csv"), index=False)

    # save vocab embeddings to pickle
    df.to_pickle(Path(dataset_dir, f"{dataset_fname.split('.')[0]}_vocab_embeddings.pkl"))

    print(f"INFO: Saved vocab embeddings. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    print(f"INFO: Total time: {round(abs((t_old:=t_start) - (t_curr:=time.time())), 3)} seconds")

if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser(description="Embed label vocabulary using OpenAI embeddings")
    parser.add_argument("--gpt_model", default="gpt-4o-mini",
                        help="GPT model used in step 02 (determines input data directory)")
    parser.add_argument("--embedding_model", default="text-embedding-3-large",
                        help="OpenAI embedding model to use")
    args = parser.parse_args()
    asyncio.run(main(args))