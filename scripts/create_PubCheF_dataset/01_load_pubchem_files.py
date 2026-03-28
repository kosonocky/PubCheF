import random
import time
import pickle as pkl
import urllib.request
from collections import defaultdict
import gzip
from pathlib import Path
from collections import Counter
import pandas as pd
import argparse

def main(args):
    t_curr = time.time()

    pubchem_data_path = Path(args.pubchem_dir)

    cid_pmid_exists = Path(pubchem_data_path, 'CID-PMID').is_file()
    cid_smiles_exists = Path(pubchem_data_path, 'CID-SMILES').is_file()

    if cid_pmid_exists and cid_smiles_exists:
        print(f"INFO: Found existing CID-PMID and CID-SMILES at {pubchem_data_path}, skipping download.\n")
    else:
        pubchem_data_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Downloading PubChem files to {pubchem_data_path}\n")

        if cid_pmid_exists:
            print("CID-PMID already exists")
        else:
            print("Downloading CID-PMID.gz")
            url = 'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-PMID.gz'
            urllib.request.urlretrieve(url, f'{pubchem_data_path}/CID-PMID.gz')
            print(f"Downloaded CID-PMID.gz. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

            with gzip.open(f'{pubchem_data_path}/CID-PMID.gz', 'rb') as f_in:
                with open(f'{pubchem_data_path}/CID-PMID', 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Unzipped CID-PMID.gz. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

        if cid_smiles_exists:
            print("CID-SMILES already exists")
        else:
            print("Downloading CID-SMILES.gz")
            url = 'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz'
            urllib.request.urlretrieve(url, f'{pubchem_data_path}/CID-SMILES.gz')
            print(f"Downloaded CID-SMILES.gz. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

            with gzip.open(f'{pubchem_data_path}/CID-SMILES.gz', 'rb') as f_in:
                with open(f'{pubchem_data_path}/CID-SMILES', 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Unzipped CID-SMILES.gz. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    create_dict = True
    if create_dict:
        # NOTE This section of code creates the CID-PMID ID Dictionary. Comment out if you have run this already and saved as pkl
        print("Loading CID to PMID Dictionary")
        cid_to_pmids = defaultdict(set)
        with open(f'{pubchem_data_path}/CID-PMID') as f:
            for line in f:
                (cid, pmid, source) = line.split()
                # currently including all sources, not separating out. If desired:
                # 1   PMIDs provided by PubChem Substance depositors
                # 2   PMIDs from the MeSH heading(s) linked to the given CID
                # 3   PMIDs provided by PubMed publishers
                # 4   PMIDs associated through BioAssays
                if source == '4':
                    cid_to_pmids[int(cid)].update([pmid]) # there are multiple PMIDs per cid

        # write dicitonary to pkl
        with open(f"{pubchem_data_path}/cid_bioassay_pmid_dict.pkl", 'wb') as f:
            pkl.dump(cid_to_pmids, f)

        print(f"Created dictionary and saved as pkl. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")
    else:
        assert Path(f'{pubchem_data_path}/cid_bioassay_pmid_dict.pkl').is_file(), "pickled CID-PMID dictionary not found"
        # tries to load pkl file
        with open(f"{pubchem_data_path}/cid_bioassay_pmid_dict.pkl", "rb") as f:
            cid_to_pmids = pkl.load(f)
        print(f"Loaded dictionary. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    count_pmids = False
    if count_pmids:
        # count the number of patents per cid and store in csv (for plotting, in order)
        pmid_counts = Counter()
        for pmid in cid_to_pmids.values():
            pmid_counts.update([len(pmid)])
        pmid_counts_df = pd.DataFrame.from_dict(pmid_counts, orient='index').reset_index()
        pmid_counts_df.columns = ["num_pmids", "num_cids"]
        # swap columns
        pmid_counts_df = pmid_counts_df[["num_cids", "num_pmids"]]
        # sort ascending by pmids
        pmid_counts_df = pmid_counts_df.sort_values(by=["num_pmids"])
        pmid_counts_df.to_csv(f"{pubchem_data_path}/num_pmids_per_cid.csv", index=False)

    # save a txt file of just the set of pmids
    pmids_set = set()
    for pmids in cid_to_pmids.values():
        pmids_set.update(pmids)
    with open(f"{pubchem_data_path}/bioassay_pmids.txt", "w") as f:
        f.write("\n".join(pmids_set))
    
    print(f"Saved bioassay_pmids.txt. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    create_rand_1k = True
    if create_rand_1k:
        # create a dictionary where for each molecule there is a maximum of 1k pmids chosen randomly
        cid_to_pmids_1k = {k: list(v) for k, v in cid_to_pmids.items()}
        for k, v in cid_to_pmids_1k.items():
            if len(v) > 1000:
                cid_to_pmids_1k[k] = list(set(random.sample(v, 1000)))
        with open(f"{pubchem_data_path}/cid_bioassay_pmid_dict_max_1k_rand.pkl", 'wb') as f:
            pkl.dump(cid_to_pmids_1k, f)
        print(f"Created dictionary with maximum 1k pmids randomly chosen if more exist. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

        # save a txt file of just the set of pmids
        pmids_set_1k = set()
        for pmids in cid_to_pmids_1k.values():
            pmids_set_1k.update(pmids)
        with open(f"{pubchem_data_path}/bioassay_pmids_max_1k_rand.txt", "w") as f:
            f.write("\n".join(pmids_set_1k))
        print(f"Saved bioassay_pmids_1k.txt. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")
    
    # Create the CID-SMILES dataframe
    print("Creating CID-SMILES dataframe")
    df = pd.read_csv(f'{pubchem_data_path}/CID-SMILES', sep='\t', header=None)
    df = df.dropna().reset_index(drop=True) # drop all rows with NaN
    df.columns = ["cid", "smiles"]
    df["cid"] = df["cid"].map(int) # convert cid to int
    df = df.set_index("cid") # make cid col the index for faster mapping. Make sure not to reset index after this point or else cid will be lost
    print(f"Loaded CID-SMILES dataframe. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    df_rand_1k = df.copy() # save a copy of the dataframe with all entries for 1k

    # Map CID to pmid dictionary to df
    df['pmids'] = df.index.map(lambda x: cid_to_pmids.get(x))
    df = df[df['pmids'] != None].dropna() # do not reset index, they represent cid
    print(f"pmids mapped to CID. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    df[["smiles", "pmids"]].to_csv(f'{pubchem_data_path}/cid_smiles_pmids.csv', index=True)
    print(f"Saved as csv. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    if create_rand_1k:
        # Map CID to pmid dictionary to df_rand_1k
        df_rand_1k['pmids'] = df_rand_1k.index.map(lambda x: cid_to_pmids_1k.get(x))
        df_rand_1k = df_rand_1k[df_rand_1k['pmids'] != None].dropna()
        print(f"1k pmids mapped to CID. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

        df_rand_1k[["smiles", "pmids"]].to_csv(f'{pubchem_data_path}/cid_smiles_pmids_max_1k_rand.csv', index=True)
        print(f"1k Saved as csv. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load PubChem CID-PMID and CID-SMILES files")
    parser.add_argument("--pubchem_dir", default=f"../../data/pubchem/{time.strftime('%Y%m%d')}",
                        help="Path to directory containing (or to download) CID-PMID and CID-SMILES. "
                             "If the files already exist there, the download is skipped.")
    args = parser.parse_args()
    main(args)