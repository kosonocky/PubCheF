
import time
from ast import literal_eval
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from collections import defaultdict
from multiprocessing import Pool, cpu_count


def process_chunk(chunk, df):
    local_mapping = defaultdict(dict)
    for _, row in chunk.iterrows():
        sub_df = df[df["labels"].apply(lambda x: bool(set(row["labels"]).intersection(x)))]
        sub_df = sub_df[sub_df["cid"] != row["cid"]]  # remove the current row

        bulk_sim = BulkTanimotoSimilarity(row["fp"], sub_df["fp"].tolist())
        sub_df["sim"] = bulk_sim

        sub_df = sub_df[sub_df["sim"] > 0.6]
        for _, sub_row in sub_df.iterrows():
            for label in row["labels"]:
                if label not in local_mapping[sub_row["cid"]] or sub_row["sim"] > local_mapping[sub_row["cid"]][label]:
                    local_mapping[sub_row["cid"]].update({label: sub_row["sim"]})
    return local_mapping

def merge_mappings(mappings):
    final_mapping = defaultdict(dict)
    for mapping in mappings:
        for cid, labels in mapping.items():
            for label, sim in labels.items():
                if label not in final_mapping[cid] or sim > final_mapping[cid][label]:
                    final_mapping[cid].update({label: sim})
    return final_mapping

def parallel_process(df, num_processes=None):
    if num_processes is None:
        num_processes = int(cpu_count()/2)

    with Pool(num_processes) as pool:
        chunksize = int(len(df) / num_processes)
        chunks = [df.iloc[i:i + chunksize] for i in range(0, df.shape[0], chunksize)]
        results = pool.starmap(process_chunk, [(chunk, df) for chunk in chunks])
        mapping = merge_mappings(results)
    return mapping

def safe_fingerprint(smiles):
    try:
        return Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    except:
        return None

def main():
    t_start = time.time()
    t_curr = t_start
    print(f"Number of cpus: {int(cpu_count()/2)}")

    data_dir = Path("../../data/final_datasets")
    fname = "cid_smiles_pmid_func_pre_prop.csv"
    df = pd.read_csv(Path(data_dir, fname)) # Load data

    df["labels"] = df["labels"].map(literal_eval)
    df["labels"] = df["labels"].map(list) # turn labels from a set into a list

    df["fp"] = df["smiles"].map(lambda x: safe_fingerprint(x))
    print(f"Number of na values in fp: {df['fp'].isna().sum()}")
    df = df.dropna(subset=["fp"])
    print(f"Number of molecules after dropping na values in fp: {len(df)}")

    print(f"Time: {time.time() - t_curr:.3f} s")


    # goal is to create mapping from target cids to source's labels
    mapping = parallel_process(df)
    print(f"Time: {time.time() - t_curr:.3f} s")
    print(f"Average time per molecule: {(time.time() - t_curr) / len(df):.3f} s")


    # this creates the entire mapping dictionary for struct sim > 0.6. As we're only using 1.0 sim, this should be rewritten to be faster as it's sloooow
    df["propagated_labels"] = df["cid"].map(lambda x: mapping[x])

    # remove keys from mapped labels if they are already in the labels
    df["propagated_labels"] = df.apply(lambda row: {k: v for k, v in row["propagated_labels"].items() if k not in row["labels"]}, axis=1)

    df[["cid", "smiles", "pmids", "labels", "propagated_labels"]].to_csv("../../data/final_datasets/cid_smiles_pmid_func_propagated.csv", index=False)
    
    cutoff = 1 # structural similarity to propagate to. 1.0 means only propagate labels that are exactly the same (still merges complexes / charge states)
    # cutoff only include labels in df["propagated_labels"] that have a confidence score of cutoff or higher
    
    print(f"{cutoff} struct similarity cutoff")
    df_cutoff = df.copy()
    df_cutoff["propagated_labels"] = df_cutoff["propagated_labels"].apply(lambda x: {k: v for k, v in x.items() if v >= cutoff})
    
    # convert labels column into a dict with values of 1
    df_cutoff["labels"] = df_cutoff["labels"].apply(lambda x: {k: 1 for k in x})

    print("min propagated label len", df_cutoff["propagated_labels"].map(len).min())
    print("avg propagated label len", df_cutoff["propagated_labels"].map(len).mean())
    print("max propagated label len", df_cutoff["propagated_labels"].map(len).max())

    df_cutoff_all_keys = []
    for i in df_cutoff["propagated_labels"]:
        df_cutoff_all_keys += [k for k in i.keys()]

    print("number of unique labels being propagated", len(set(df_cutoff_all_keys)))
    print("total number of labels being propagated", len(df_cutoff_all_keys))
    print("number of molecules with at least one propagated label", len(df_cutoff[df_cutoff["propagated_labels"].map(len) > 0]))
    

    # merge with propagated_labels column
    df_cutoff["labels"] = df_cutoff.apply(lambda x: {**x["labels"], **x["propagated_labels"]}, axis=1)
    df_cutoff = df_cutoff.drop(columns=["propagated_labels"])
    # df_cutoff.to_csv(f"../../data/final_datasets/cid_smiles_pmid_func_propagated_cutoff_{cutoff}_soft.csv", index=False)
    # print("soft label df saved")

    # convert labels column into a list of keys
    df_cutoff["labels"] = df_cutoff["labels"].apply(lambda x: list(x.keys()))
    df_cutoff.to_csv(f"../../data/final_datasets/cid_smiles_pmid_func_propagated_cutoff_{cutoff}_hard.csv", index=False)

    print("hard label df saved")

    print()




if __name__ == "__main__":
    main()