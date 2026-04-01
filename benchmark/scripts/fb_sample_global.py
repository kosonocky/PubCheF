import time
import sys
sys.setrecursionlimit(12000)
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from rdkit import DataStructs
from rdkit import Chem


# global variable, beware!
ds_df = pd.DataFrame()

def rec_feature_balance_df_fp(fb_df=None, cutoff:float = 0.0, feat_col_name:str = "fp", count = 0, start_time = time.time()):
    global ds_df
    count = count + 1
    
    if count >= 10000 or len(ds_df) == 0:
        # recurse out of function to avoid recursion limit / finish feature balancing
        return fb_df

    if len(fb_df) == 1:
        ds_df["sim"] = DataStructs.BulkTanimotoSimilarity(fb_df[feat_col_name].iloc[0], ds_df[feat_col_name].tolist())
    else:
        ds_df["sim"] = ds_df.apply(lambda x: np.max(DataStructs.BulkTanimotoSimilarity(x[feat_col_name], fb_df[feat_col_name].tolist())), axis=1)

    ds_df = ds_df[ds_df["sim"] < cutoff].reset_index(drop=True) # drop any rows w values above our cutoff value
    
    if count % 500 == 0:
        print(f"Count: {count}. {round(time.time() - start_time, 3)} seconds elapsed since start. ds_df length: {len(ds_df)}")
    
    ds_df = ds_df.sort_values(by="sim",ascending=False).reset_index(drop=True)
    ds_df_0 = ds_df.iloc[0:1]
    ds_df = ds_df.iloc[1:]
    fb_df = pd.concat([fb_df,rec_feature_balance_df_fp(fb_df=ds_df_0,                        
                                                    cutoff=cutoff,
                                                    feat_col_name=feat_col_name,
                                                    start_time=start_time,
                                                    count = count
                                                    )])        
    return fb_df

def safe_fingerprint(smiles):
    try:
        return Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    except:
        return None


def main():
    print(f"Cutoff: {(cutoff:=0.5)}")
    
    start_time = time.time()
    
    global ds_df # need this as global var
    ds_df = pd.read_pickle("../../data/final_datasets/preprocessed_propagated_1_hard/scaffold_split/test.csv")
    ds_df["fp"] = ds_df["smiles"].apply(safe_fingerprint)
    print(ds_df.columns)

    fb_df = None
    # fb_df = None # replace with pd.read pickle of fb_df if supplementing
    if isinstance(fb_df, type(None)):
        print("fb_df length is 0. Setting fb_df to first value of ds_df.")
        fb_df = ds_df.iloc[0:1]
        ds_df = ds_df.iloc[1:]

    print("loaded data")

    # until feature-balancing is done, rerun every 10000 recursions
    while len(ds_df) != 0:
        print("len(ds_df)", len(ds_df))
        print("len(fb_df)", len(fb_df))
        print("\nPreparing to feature balance")
        fb_df = rec_feature_balance_df_fp(fb_df=fb_df,cutoff=cutoff,feat_col_name="fingerprint", start_time = start_time).reset_index(drop=True)


    print("recursion done.")
    print("fbdf length", len(fb_df))

    fb_df.to_csv(f"chef-v2_test_fp_balanced_{cutoff}.csv")
    fb_df.to_pickle(f"chef-v2_test_fp_balanced_{cutoff}.pkl")
    
    print(f"{time.time() - start_time} seconds total elapsed.")
    print("done.")        

if __name__ == "__main__":
    main()