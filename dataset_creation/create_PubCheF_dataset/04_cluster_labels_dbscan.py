import time
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
from ast import literal_eval
import multiprocessing as mp
import argparse

def main(args):
    # metric = "cosine"
    metric = "euclidean"
    t_start = time.time()
    t_curr = t_start
    vocab_embedding_dir = f"../../data/dataset_creation/bioassay_pmids/{args.gpt_model}"
    vocab_embedding_fname = "pmid_func_complete_vocab_embeddings.pkl"
    save_dir = Path(vocab_embedding_dir, "label_consolidation")
    save_dir.mkdir(parents = True, exist_ok = True)
    df = pd.read_pickle(Path(vocab_embedding_dir, vocab_embedding_fname))
    print(f"INFO: Embedding data loaded from {Path(vocab_embedding_dir, vocab_embedding_fname)}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    arr = np.array(df['embedding'].tolist())

    eps_values = np.arange(0.2, 0.8, 0.005) if args.scan_eps else [args.eps]

    for eps in eps_values:
        print(f"INFO: eps = {eps}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
        dbscan = DBSCAN(eps = eps, min_samples = 1, n_jobs=mp.cpu_count()//2, metric=metric)
        dbscan.fit(arr)
        dbscan_df = df[["text"]]
        dbscan_df['dbscan'] = dbscan.labels_
        print(f"INFO: {len(dbscan_df['dbscan'].unique())} clusters obtained. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

        # convert each cluster to a set, then add to a set of sets
        all_clusters_set = set()
        for cluster in dbscan_df['dbscan'].unique():
            all_clusters_set.add(frozenset(dbscan_df[dbscan_df['dbscan'] == cluster]['text'].tolist()))

        # convet to list of lists and sort by list length        
        all_clusters_list = [sorted(list(x)) for x in all_clusters_set]
        all_clusters_list.sort(key=len, reverse=True)

        with open(Path(save_dir, f"dbscan_{metric}_minsamples_1_eps_{eps:.3f}.txt"), "w") as f:
            for item in all_clusters_list:
                f.write(f"{item}\n")
    
    print(f"INFO: Clustering complete. Total time: {round(abs((t_start:=t_curr) - (t_curr:=time.time())), 3)} seconds")

if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser(description="Cluster label vocabulary with DBSCAN")
    parser.add_argument("--gpt_model", default="gpt-4o-mini",
                        help="GPT model used in step 02 (determines input data directory)")
    parser.add_argument("--eps", type=float, default=0.41,
                        help="DBSCAN epsilon value (default: 0.41)")
    parser.add_argument("--scan_eps", action="store_true",
                        help="Scan eps from 0.2 to 0.8 in steps of 0.005 (development mode)")
    args = parser.parse_args()
    main(args)