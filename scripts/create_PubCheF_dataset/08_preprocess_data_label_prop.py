import time
from ast import literal_eval
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tdc.utils.split import create_scaffold_split
from collections import Counter
import pickle as pkl

from transformers import AutoTokenizer
from tokenizers.models import WordLevel
from tokenizers import Regex
from tokenizers import Tokenizer, Regex
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing

# suppress chem warnings/errors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def canon_smiles(smiles):
    try: 
        return Chem.CanonSmiles(smiles, useChiral = True)
    except Exception as e:
        print(e)
        print(f"ERROR: {smiles} set to None")
        return None
    
def remove_long_smiles(smiles, tokenizer):
    """
    Remove SMILES that tokenize to >= 510 tokens
    """
    try:
        tokenized_output = tokenizer([smiles])
        if len(tokenized_output['input_ids'][0]) >= 510:
            return None
        else:
            return smiles
    except Exception as e:
        print(f"ERROR: {smiles} set to None")
        return None
    
def fingerprint(smiles):
    try:
        return Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    except:
        return None
    

def drop_uncommon_labels(df, threshold=50, label_type="hard"):
    counter = Counter()
    # create set of all labels # NOTE This is done twice (see create_mlb) and should be consolidated to just once
    if label_type == "hard":
        for labels in df['labels']:
            counter.update(labels)
    elif label_type in ["soft", "soft_squared"]:
        for labels in df["labels"]:
            counter.update(labels.keys())
    else:
        raise ValueError(f"ERROR: label_type must be 'hard', 'soft', or 'soft_squared', not {label_type}")

    # list of most common keys
    labels_threshold = [label for label, count in counter.most_common() if count >= threshold]
    labels_100 = [label for label, count in counter.most_common() if count >= 100]
    labels_50 = [label for label, count in counter.most_common() if count >= 50]
    labels_20 = [label for label, count in counter.most_common() if count >= 20]
    labels_10 = [label for label, count in counter.most_common() if count >= 10]
    labels_5 = [label for label, count in counter.most_common() if count >= 5]


    print(f"INFO: At a threshold of 5, we would keep {len(labels_5)} labels out of {len(counter)} labels")
    print(f"INFO: At a threshold of 10, we would keep {len(labels_10)} labels out of {len(counter)} labels")
    print(f"INFO: At a threshold of 20, we would keep {len(labels_20)} labels out of {len(counter)} labels")
    print(f"INFO: At a threshold of 50, we would keep {len(labels_50)} labels out of {len(counter)} labels")
    print(f"INFO: At a threshold of 100, we would keep {len(labels_100)} labels out of {len(counter)} labels")

    print(f"INFO: The threshold is {threshold}, we are keeping {len(labels_threshold)} labels out of {len(counter)} labels")

    # drop labels in set if not in all_labels
    if label_type == "hard":
        df['labels'] = df['labels'].apply(lambda x: set(x).intersection(labels_threshold))
    elif label_type in ["soft", "soft_squared"]:
        df['labels'] = df['labels'].apply(lambda x: {k: v for k, v in x.items() if k in labels_threshold})

    return df


def create_mlb(df, label_type="hard", save_dir="."):
    # create set of all labels
    all_labels = Counter()
    if label_type == "hard":
        for labels in df['labels']:
            all_labels.update(labels)
    elif label_type in ["soft", "soft_squared"]:
        for labels in df["labels"]:
            all_labels.update(labels.keys())
    else:
        raise ValueError(f"ERROR: label_type must be 'hard', 'soft', or 'soft_squared', not {label_type}")

    # list of most common keys
    all_labels = [label for label, _ in all_labels.most_common()]

    n_samples = df.shape[0]
    n_classes = len(all_labels)
    print(f"INFO: n_samples: {n_samples}, n_classes: {n_classes}")

    # create multi-label binary matrix
    mlb = MultiLabelBinarizer(classes=all_labels)
    if label_type == "hard":
        mlb = mlb.fit(df['labels'])
    elif label_type in ["soft", "soft_squared"]:
        mlb = mlb.fit(df['labels'].apply(lambda x: x.keys()))

    # save mlb
    with open(Path(save_dir, "mlb.pkl"), "wb") as f:
        pkl.dump(mlb, f)

class SMILESTokenizer:
    """
    Tokenizer for SMILES strings. AutoTokenizer using DeepChem/ChemBERTa-77M-MLM was broken and this fixes it.
    Requires the vocab.json file from the DeepChem/ChemBERTa-77M-MLM model.
    """

    def __init__(self, vocab_path="tokenizer/vocab.json", download_vocab=True):
        if download_vocab:
            tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            Path("tokenizer").mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained("tokenizer")

        self.tokenizer = Tokenizer(
            WordLevel.from_file(
                vocab_path, 
                unk_token='[UNK]'
            )
        )
        self.pre_tokenizer = Split(
            pattern=Regex("Cl|Br|%[0-9]{2}|>>|\[(.*?)\]|."),
            behavior='isolated'
        )
        self.tokenizer.pre_tokenizer = self.pre_tokenizer
        self.tokenizer.add_special_tokens(["[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )
    
    # make this the default method for the class
    def __call__(self, text):
        """
        Encode a list of SMILES strings
        """
        encoded_corpus = [self.tokenizer.encode(t) for t in text]

        max_len = max((len(encoded.ids) for encoded in encoded_corpus))
        padded_corpus = np.zeros((len(encoded_corpus), max_len), dtype=np.int64)
        attention_masks = np.zeros_like(padded_corpus)

        for i, encoded in enumerate(encoded_corpus):
            padded_corpus[i, : len(encoded.ids)] = encoded.ids
            attention_masks[i, : len(encoded.ids)] = 1

        return {"input_ids": padded_corpus, "attention_mask": attention_masks}


def main():
    t_start = time.time()
    t_curr = t_start

    data_dir = Path("../../data/final_datasets")
    
    # for fp_cutoff in [0.6, 0.7, 0.8, 0.85, 0.9, 1]:
    #     for label_type in ["soft", "soft_squared", "hard"]:
    fp_cutoff = 1
    label_type = "hard"

    print(f"\nINFO: Processing data with fp_cutoff: {fp_cutoff} and label_type: {label_type}")
    fname = f"cid_smiles_pmid_func_propagated_cutoff_{fp_cutoff}_{label_type}.csv"
    
    df = pd.read_csv(Path(data_dir, fname)) # Load data

    save_dir = Path(data_dir, f"preprocessed_propagated_{fp_cutoff}_{label_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    df["labels"] = df["labels"].map(literal_eval)

    print(f"INFO: Data loaded from {Path(data_dir, fname)}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    print(f"INFO: There are {df.shape[0]} rows")

    df = df.set_index("smiles")     # set smiles to be index (makes it faster to canonicalize smiles)
    df["smiles"] = df.index.map(canon_smiles)    # canonicalize smiles
    df = df.reset_index(drop=True) # bring smiles back as a column
    df = df.dropna()
    df = df[df["smiles"] != "None"]
    print(f"INFO: smiles canonicalized, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    print(f"INFO: There are now {df.shape[0]} rows")
    
    if label_type == "hard":
        df = df.groupby("smiles", as_index=False).agg({"labels": "sum", "cid": "first"}) # drop duplicates based on smiles. Merge in labels for the dropped rows into the first row
    elif label_type in ["soft", "soft_squared"]:
        def merge_dicts(series):
            result = {}
            for d in series:
                result.update(d)
            return result
        df = df.groupby("smiles", as_index=False).agg({"labels": merge_dicts, "cid": "first"})
    else:
        raise ValueError(f"ERROR: label_type must be 'hard', 'soft', or 'soft_squared', not {label_type}")
    
    print(f"INFO: Duplicates merged, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    print(f"INFO: There are now {df.shape[0]} rows")
    
    # ChemBERTa has a context length of 510 tokens (including 2 for special tokens)
    df = df.set_index("smiles")     # set smiles to be index (makes it faster to canonicalize smiles)
    tokenizer = SMILESTokenizer()
    df["smiles"] = df.index.map(lambda x: remove_long_smiles(x, tokenizer))
    df = df.dropna()
    df = df.reset_index(drop=True) # remove smiles with value None. I'm sure you can do this on the index but just to be safe
    print(f"INFO: There are now {df.shape[0]} rows after dropping smiles that tokenize >= 510 tokens")

    df = df.set_index("smiles")
    df["fingerprint"] = df.index.map(fingerprint) # generate fingerprints
    print(f"INFO: Fingerprints generated, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # generate scaffolds (not necessary but useful for later tracking)
    df["scaffold"] = df.index.map(MurckoScaffold.MurckoScaffoldSmiles) # generate scaffolds
    df = df.reset_index()
    print(f"INFO: Scaffolds generated, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # drop uncommon labels
    df = drop_uncommon_labels(df, threshold=50, label_type=label_type)
    print(f"INFO: Uncommon labels dropped, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    df = df[df["labels"].map(len) > 0] # remove rows with no labels
    print(f"INFO: Rows with no labels removed, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    print(f"INFO: There are now {df.shape[0]} rows")

    # create mlb
    create_mlb(df, label_type=label_type, save_dir=save_dir)
    print(f"INFO: MultiLabelBinarizer created and saved, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    df = df[["cid", "smiles", "scaffold", "labels", "fingerprint"]]
    df.to_pickle(Path(save_dir, f"{fname[:-4]}_preprocessed.pkl"))
    print(f"INFO: Data saved to {Path(save_dir, f'{fname[:-4]}_preprocessed.pkl')}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # train validation test splits (80/10/10) (random split)
    print(f"INFO: Splitting into train, val, and test sets based on random split")
    df_train_random, df_test_random = train_test_split(df, test_size=0.2, random_state=42)
    df_val_random, df_test_random = train_test_split(df_test_random, test_size=0.5, random_state=42)

    # create folder for random splits
    save_dir_random = Path(save_dir, "random_split")
    save_dir_random.mkdir(parents=True, exist_ok=True)

    df_train_random.to_pickle(Path(save_dir_random, "train_fp.pkl"))
    df_train_random[["cid", "smiles", "scaffold", "labels"]].to_csv(Path(save_dir_random, "train.csv"), index=False)
    df_val_random.to_pickle(Path(save_dir_random, "val_fp.pkl"))
    df_val_random[["cid", "smiles", "scaffold", "labels"]].to_csv(Path(save_dir_random, "val.csv"), index=False)
    df_test_random.to_pickle(Path(save_dir_random, "test_fp.pkl"))
    df_test_random[["cid", "smiles", "scaffold", "labels"]].to_csv(Path(save_dir_random, "test.csv"), index=False)

    # split based on scaffolds
    split_data = create_scaffold_split(df,
        seed = 42,
        frac=[0.8, 0.1, 0.1],
        entity="smiles",
    )
    df_train_scaffold = split_data["train"]
    df_val_scaffold = split_data["valid"]
    df_test_scaffold = split_data["test"]

    # create folder for scaffold splits and save
    save_dir_scaffold = Path(save_dir, "scaffold_split")
    save_dir_scaffold.mkdir(parents=True, exist_ok=True)
    df_train_scaffold.to_pickle(Path(save_dir_scaffold, "train_fp.pkl"))
    df_train_scaffold[["cid", "smiles", "scaffold", "labels"]].to_csv(Path(save_dir_scaffold, "train.csv"), index=False)
    df_val_scaffold.to_pickle(Path(save_dir_scaffold, "val_fp.pkl"))
    df_val_scaffold[["cid", "smiles", "scaffold", "labels"]].to_csv(Path(save_dir_scaffold, "val.csv"), index=False)
    df_test_scaffold.to_pickle(Path(save_dir_scaffold, "test_fp.pkl"))
    df_test_scaffold[["cid", "smiles", "scaffold", "labels"]].to_csv(Path(save_dir_scaffold, "test.csv"), index=False)

    print(f"INFO: Total time: {round(abs((time.time()) - (t_start)), 3)} seconds")


if __name__ == "__main__":
    print(__file__)
    main()