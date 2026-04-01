import time
import pandas as pd
from pathlib import Path
from ast import literal_eval
from collections import Counter
import numpy as np
import argparse


def get_bad_labels():
    """
    Returns a list of bad labels that we want to remove from the dataset
    """
    return ["NO INFO",
            "NO_FUNCTIONAL_DESCRIPTORS",
            "NO_CHEMICAL_FUNCTIONAL_DESCRIPTORS",
            "NO_FUNCTIONAL_INFORMATION",
            "NO Function Descriptors",
            "NO Pharmacological Function",
            "NO_MOLECULE_FUNCTION",
            "NO CATEGORIES FOUND",
            "NO Descriptors Found",
            "NO INFO",
            "NO CATEGORIES",
            "NO Biological Activity",
            "NO Function Descriptors",
            "NO Pharmacological Function",
            "NO_MOLECULE_DESCRIPTORS",
            "NO_CHEMICAL_INFO",
            "NO_ABSTRACT_AVAILABLE",
            "NO_TITLE",
            "NO_ACTIVITY",
            "NO_INFO",
            "NO_INFORMATION",
            "NO_MOLECULES_DETECTED",
            "NO_Summary_Available",
            "NO_ABSTRACT",
            "NO_CHEMICAL_INFORMATION",
            ]

def main(args):
    t_start = time.time()
    t_curr = t_start
    data_dir = Path(f"../../data/dataset_creation/bioassay_pmids/{args.gpt_model}")
    dataset_df = pd.read_csv(Path(data_dir, "pmid_func_complete_formatted.csv"))
    dataset_df['labels'] = dataset_df['labels'].apply(lambda x: literal_eval(x))
    print(f"INFO: Dataset loaded, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    embeddings_df = pd.read_pickle(Path(data_dir, "pmid_func_complete_vocab_embeddings.pkl"))
    embeddings_df["embedding"] = embeddings_df["embedding"].map(lambda x: np.array(x))
    print(f"INFO: Embeddings loaded, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # load clusters
    with open(Path(data_dir, "label_consolidation", f"dbscan_euclidean_minsamples_1_eps_{args.eps:.3f}.txt"), "r") as f:
        clusters = [literal_eval(line) for line in f.readlines()]
    
    # for each cluster, find the centroid embedding and make that 'map_to'. make 'map_from' each word in that cluster
    map_df = pd.DataFrame(columns = ["word_from", "word_to", "cluster"])
    for cluster in clusters:
        if len(cluster) == 1:
            map_df = pd.concat([map_df, pd.DataFrame({"word_from": cluster, "word_to": cluster, "cluster": [cluster]})])
            continue
        cluster_df = embeddings_df[embeddings_df["text"].isin(cluster)]
        arr = np.array(cluster_df["embedding"].tolist())
        mean_embedding = arr.mean(0)
        closest_idx = np.argmin(np.linalg.norm(arr - mean_embedding, axis = 1))
        closest_word = cluster_df.iloc[closest_idx]["text"]

        # map each word in cluster to the representative word
        for word in cluster:
            map_df = pd.concat([map_df, pd.DataFrame({"word_from": [word.lower()], "word_to": [closest_word], "cluster": [cluster]})])

    # print number of unique words in map_df['map_to']
    print(f"INFO: Number of unique words in map_df['word_to']: {len(map_df['word_to'].unique())}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    # for each word in map_df['word_to'], if there is another word that shares the same alphanumeric string components as another 'word_to', set 'word_to' to the shorter word
    map_df["word_to_tmp"] = map_df["word_to"].map(lambda x: x.lower())
    map_df["word_to_tmp"] = map_df["word_to_tmp"].map(lambda x: "".join([char for char in x if char.isalnum()]))
    # create word_to word_to mapping that maps based on shared word_to_tmp (mapping to the shortest word_to)
    word_to_dict = dict()
    for _, row in map_df.iterrows():
        if row["word_to_tmp"] in word_to_dict:
            if len(row["word_to"]) < len(word_to_dict[row["word_to_tmp"]]):
                word_to_dict[row["word_to_tmp"]] = row["word_to"]
        else:
            word_to_dict[row["word_to_tmp"]] = row["word_to"]
    map_df["word_to"] = map_df["word_to_tmp"].map(lambda x: word_to_dict[x])
    map_df = map_df.drop(columns = ["word_to_tmp"])
    print(f"INFO: After merging shared alphanumeric strings, number of unique words in map_df['word_to']: {len(map_df['word_to'].unique())}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # save map_df to csv
    map_df.to_csv(Path(data_dir, "pmid_func_complete_formatted_cluster_map.csv"), index = False)
    print(f"INFO: Cluster map saved, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    
    map_dict = dict(zip(map_df["word_from"], map_df["word_to"]))

    # # for each word in dataset_df['labels'] (which is a list of multi-word strings), map to the corresponding word_to in map_df if the word is in map_from
    def map_labels(x):
        return [map_dict.get(word.lower(), word) for word in x]
    
    dataset_df['labels'] = dataset_df['labels'].apply(map_labels)
    
    print(f"INFO: Labels mapped, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # remove bad labels
    bad_labels = get_bad_labels()
    dataset_df['labels'] = dataset_df['labels'].apply(lambda x: [word for word in x if word not in bad_labels])
    print(f"INFO: Removed bad labels from dataset_df, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    dataset_df['labels'] = dataset_df['labels'].apply(lambda x: list(set(x))) # remove duplicates
    print(f"INFO: Removed duplicates, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    cid_pmid = pd.read_csv(Path(args.pubchem_dir, "CID-PMID"), sep="\t", header=None)
    cid_pmid.columns = ["cid", "pmid", "source"]
    cid_pmid_4 = cid_pmid[cid_pmid["source"] == 4] # only pubmed bioassay
    pmid_counter_4 = Counter(cid_pmid_4["pmid"]) # just pmids from source 4 (bioassay), as our dataset is from source 4
    with open(Path(data_dir, "bioassay_pmid_cid_counter.txt"), "w") as f:
        for pmid, count in pmid_counter_4.most_common():
            f.write(f"{pmid}: {count}\n")
    # remove pmids with more than 100 cids. These are likely to be false positives
    # NOTE This should probably go much earlier in the pipeline. Probably at 01_load_pubchem_files.py
    pmids_to_remove = []
    for pmid, count in pmid_counter_4.most_common():
        if count > 100:
            pmids_to_remove.append(pmid)
    print(f"INFO: Number of pmids to remove: {len(pmids_to_remove)}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    dataset_df = dataset_df[~dataset_df["pmid"].isin(pmids_to_remove)]
    print(f"INFO: Removed pmids with more than 100 cids, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # save to csv
    dataset_df.to_csv(Path(data_dir, "pmid_func_complete_formatted_mapped.csv"), index = False)
    print(f"INFO: Data saved, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # get vocabulary from labels
    vocab = set([item for sublist in dataset_df['labels'].tolist() for item in sublist])

    # save new vocabulary to txt
    with open(Path(data_dir, "pmid_func_complete_formatted_mapped_vocab.txt"), "w") as f:
        for item in sorted(vocab):
            f.write(f"{item}\n")
    print(f"INFO: Vocabulary saved, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    print(f"INFO: Total time: {round(abs((t_start:=t_curr) - (t_curr:=time.time())), 3)} seconds")

if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser(description="Map GPT labels to CID-SMILES dataset")
    parser.add_argument("--gpt_model", default="gpt-4o-mini",
                        help="GPT model used in step 02 (determines input data directory)")
    parser.add_argument("--eps", type=float, default=0.41,
                        help="DBSCAN epsilon used in step 04 (must match)")
    parser.add_argument("--pubchem_dir", default=f"../../data/pubchem/{time.strftime('%Y%m%d')}",
                        help="Path to directory containing CID-PMID and CID-SMILES")
    args = parser.parse_args()
    main(args)