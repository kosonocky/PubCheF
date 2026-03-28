import time
import pandas as pd
from pathlib import Path
from ast import literal_eval
from collections import defaultdict, Counter
import argparse


def main(args):
    t_start = time.time()
    t_curr = t_start
    data_dir = Path(f"../../data/dataset_creation/bioassay_pmids/{args.gpt_model}")
    dataset_df = pd.read_csv(Path(data_dir, "pmid_func_complete_formatted_mapped.csv"))
    dataset_df = dataset_df[["pmid", "labels"]]
    dataset_df["labels"] = dataset_df["labels"].map(literal_eval)
    print(f"INFO: PMID Func data loaded, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    cid_smiles_pmid_df = pd.read_csv(Path(args.pubchem_dir, "cid_smiles_pmids_max_1k_rand.csv"))
    cid_smiles_pmid_df["pmids"] = cid_smiles_pmid_df["pmids"].map(literal_eval)
    # convert to int
    cid_smiles_pmid_df["pmids"] = cid_smiles_pmid_df["pmids"].map(lambda x: [int(y) for y in x])
    
    print(f"INFO: CID SMILES data loaded, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # convert dataset_df into a dict
    dataset_dict = {}
    for _, row in dataset_df.iterrows():
        dataset_dict[row["pmid"]] = row["labels"]

    # for each pmid in pmids, add the corresponding 'labels' if it exists in dataset_dict
    cid_smiles_pmid_df['labels'] = cid_smiles_pmid_df['pmids'].map(lambda x: [dataset_dict.get(y, []) for y in x])


    print(f"INFO: Mapped labels, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # collapse into single list
    cid_smiles_pmid_df['labels'] = cid_smiles_pmid_df['labels'].map(lambda x: [y for z in x for y in z])

    # convert to counter and only take the top 30 labels
    cid_smiles_pmid_df['labels'] = cid_smiles_pmid_df['labels'].map(lambda x: [y[0] for y in Counter(x).most_common(30)])
    
    # remove rows with empty sets
    cid_smiles_pmid_df = cid_smiles_pmid_df[cid_smiles_pmid_df['labels'].map(len) > 0]

    # save
    save_dir = Path("../../data/final_datasets")
    save_dir.mkdir(parents=True, exist_ok=True)
    cid_smiles_pmid_df.to_csv(Path(save_dir, "cid_smiles_pmid_func_pre_prop.csv"), index = False)
    print(f"INFO: Data saved. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    print(f"INFO: Total time: {round(abs((t_curr) - (t_start)), 3)} seconds")

if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser(description="Create CID-SMILES functional dataset")
    parser.add_argument("--gpt_model", default="gpt-4o-mini",
                        help="GPT model used in step 02 (determines input data directory)")
    parser.add_argument("--pubchem_dir", default=f"../../data/pubchem/{time.strftime('%Y%m%d')}",
                        help="Path to directory containing PubChem files (output of step 01)")
    args = parser.parse_args()
    main(args)