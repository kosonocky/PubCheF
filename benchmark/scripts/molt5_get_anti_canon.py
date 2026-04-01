"""
MolT5 was trained on the SMILES in PubChem which are different than those produced by RDKit. So this script, and molt5_apply_anti_canon.ipynb, are used to convert the RDKit SMILES to the PubChem SMILES. This is done by creating a mapping from the RDKit canonical SMILES to the PubChem SMILES using a subset of the PubChem data. The mapping is then saved as a JSON file which can be used to convert the RDKit SMILES to the PubChem SMILES.
"""

import pandas as pd
import json

from rdkit import Chem

def canon_smiles(smiles, isomericSmiles=True, kekuleSmiles=False):
    try: 
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles)
    except Exception as e:
        print(e)
        print(f"ERROR: {smiles} set to None")
        return None





def main():
    df = pd.read_csv(f'../../data/pubchem/20240111/CID-SMILES', sep='\t', header=None, nrows=100000)
    df = df.dropna().reset_index(drop=True) # drop all rows with NaN
    df.columns = ["cid", "smiles"]
    df = df.drop(columns=["cid"])
    df = df.set_index("smiles")
    df["canon_smiles"] = df.index.map(canon_smiles)
    # create dict to convert canon to no_canon
    canon_to_no_canon = df["canon_smiles"].to_dict()

    # reverse dict
    canon_to_no_canon = {v: k for k, v in canon_to_no_canon.items()}
    
    # for k, v in canon_to_no_canon.items():
    #     print(k, v)

    # save dict
    with open(f'rdkit_to_pubchem_canon.json', 'w') as f:
        json.dump(canon_to_no_canon, f)


if __name__ == "__main__":
    main()