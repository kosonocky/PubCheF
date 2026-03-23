import os
import time
import random
import pickle as pkl
import argparse
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
)
from tokenizers.models import WordLevel
from tokenizers import Regex
from tokenizers import Tokenizer, Regex
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from rdkit import Chem


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# keeping off because it slows down training
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

os.environ['RDMAV_HUGEPAGES_SAFE'] = '1' # this is needed to avoid a warning from transformers

class ChemMLMRegressor(nn.Module):
    """
    Classification on ChemBERTa
    """

    def __init__(
        self,
        model_name="DeepChem/ChemBERTa-77M-MLM",
        model_initialization="pretrained",
        drop_rate=0.2,
        d_out=1,
    ):
        super(ChemMLMRegressor, self).__init__()
        if model_initialization == "pretrained":
            self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            self.bert = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(model_name))

        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.bert.config.hidden_size, d_out),
        )

    def forward(self, input_ids, attention_masks, extract_embeddings=False):
        """
        Called when model object is called
        """
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True
        )
        if extract_embeddings:
            return {"embedding": outputs["hidden_states"][-1][:, 0, :]}
        
        predictions = self.regressor(outputs["hidden_states"][-1][:, 0, :])  # state of cls token
        return {"predictions": predictions}
    
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
    def __call__(self, text, length_cutoff=510):
        """
        Encode a list of SMILES strings

        Pads to longest SMILES string in the list, or to length_cutoff, whichever is shorter.
        """
        encoded_corpus = [self.tokenizer.encode(t) for t in text]

        max_len = max((len(encoded.ids) for encoded in encoded_corpus))
        max_len = min(max_len, length_cutoff)

        padded_corpus = np.zeros((len(encoded_corpus), max_len), dtype=np.int64)
        attention_masks = np.zeros_like(padded_corpus)

        for i, encoded in enumerate(encoded_corpus):
            padded_corpus[i, : len(encoded.ids[:max_len])] = encoded.ids[:max_len]
            attention_masks[i, : len(encoded.ids[:max_len])] = 1

        return {"input_ids": padded_corpus, "attention_mask": attention_masks}
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # high alpha means more weight to positive class. Low alpha means more weight to negative class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.float()
        if self.alpha is not None:
            at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            at = 1
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Noam learning rate scheduler. This is a custom scheduler that is not available in PyTorch.
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            (self.d_model ** -0.5) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
            for base_lr in self.base_lrs
        ]
    
def canon_smiles_single_random(smiles, isomericSmiles=True):
    """
    Canonicalize a SMILES string by choosing a random atom as the root. Works for each part of a multi-molecule SMILES string complex
    """
    try:
        final_str = ""
        for s in smiles.split("."):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                final_str += s
                final_str += "."
                continue
            atom_count = sum([1 for atom in mol.GetAtoms()])
            random_choice = random.choice(range(atom_count))
            final_str += Chem.MolToSmiles(mol, canonical=True, rootedAtAtom=random_choice, isomericSmiles=isomericSmiles)
            final_str += "."
        final_str = final_str[:-1]
        return final_str
    except Exception as e:
        print(f"Triggered exception: {e} for smiles: {smiles}. Returning original smiles.")
        return smiles
    
def canon_smiles(smiles, isomericSmiles=True, kekuleSmiles=False):
    try: 
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles)
    except Exception as e:
        print(e)
        print(f"ERROR: {smiles} set to None")
        return None

def smooth_labels(y, alpha=0.1):
    """
    Label smoothing
    """
    with torch.no_grad():
        y = y * (1 - alpha) + (alpha / y.shape[1])
    return y

def load_model(
    model_name = "DeepChem/ChemBERTa-77M-MLM",
    model_initialization = "pretrained",
    drop_rate = 0.2,
    d_out = 1,
):
    model = ChemMLMRegressor(
        model_name=model_name,
        model_initialization=model_initialization,
        drop_rate=drop_rate, 
        d_out = d_out,
    )
    d_model = model.bert.config.hidden_size
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU.")

        # TODO Need to use better parallel method. This is not super efficient from what I know
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)
    return model, device, d_model


def create_dataloader(X, tokenizer, batch_size=32, seed=True, shuffle=True):

    # tokenize data, get masks, and create dataloaders
    encoded_corpus = tokenizer(
        text=X.tolist(),
    )

    inputs = np.array(encoded_corpus["input_ids"])
    masks = np.array(encoded_corpus["attention_mask"])
    masks = masks.astype(bool)

    # create dataloaders
    dataset = TensorDataset(
        torch.tensor(inputs),
        torch.tensor(masks),
    )

    generator = torch.Generator().manual_seed(seed)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
    )

    return dataloader


def get_probs_from_model(
        model,
        X_smiles,
        tokenizer,
        device,
        t_curr,
        batch_size=256,
        seed=42,
        extract_embeddings=False,
        ):

    X_smiles = np.array(X_smiles)
    test_dataloader = create_dataloader(
        X_smiles, tokenizer, batch_size=batch_size, seed=seed, shuffle=False,
    )

    all_predictions = []
    model.eval()
    with torch.no_grad():
        for count, batch in enumerate(test_dataloader):
            if count % 500 == 0:
                print(f"INFO: Batch {count} of {len(test_dataloader)}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            outputs = model(b_input_ids, b_input_mask, extract_embeddings=extract_embeddings)
            if extract_embeddings:
                predictions = outputs["embedding"]
                predictions = predictions.cpu().numpy()
            else:
                predictions = outputs["predictions"]
                # convert predictions and labels to numpy arrays
                predictions = predictions.cpu().numpy()
                # convert logits to probabilities
                predictions = 1 / (1 + np.exp(-predictions))
            
            all_predictions.append(predictions)

    all_predictions = np.concatenate(all_predictions)

    return all_predictions

def main(args):
    t_start = time.time()
    t_curr = t_start
    isomericSmiles = args.isomericSmiles
    mlb_dir = Path(args.mlb_dir)
    model_dir = Path(args.model_dir)

    # unpack args
    if args.input_csv:
        input_name = Path(args.input_csv).stem
        input_df = pd.read_csv(args.input_csv)
        if args.smiles_column not in input_df.columns:
            raise ValueError(f"SMILES column '{args.smiles_column}' not found in CSV.")
            
        input_df["canon_smiles"] = input_df[args.smiles_column].apply(lambda x: canon_smiles(x, isomericSmiles=isomericSmiles))
        input_df = input_df.dropna(subset=["canon_smiles"]).reset_index(drop=True)
        X_smiles = input_df["canon_smiles"].tolist()
    else:
        input_name = "single_smiles"
        X_smiles = [canon_smiles(args.smiles, isomericSmiles=isomericSmiles)]
        
        none_indices = [i for i, smiles in enumerate(X_smiles) if smiles is None]
        X_smiles = [smiles for i, smiles in enumerate(X_smiles) if i not in none_indices]

    print(f"\n===Input Params===")
    print(f"INFO: Input target: {args.input_csv or args.smiles}")
    print(f"INFO: p_threshold: {args.p_threshold}")
    print(f"INFO: batch_size_chunk: {args.batch_size}")
    print(f"INFO: extract_embeddings: {args.extract_embeddings}")
    print(f"\n===Training Params===")
    print(f"INFO: batch_size: {(batch_size:=256)}")
    print(f"INFO: gpus available: {torch.cuda.device_count()}")
    print(f"INFO: model_name: {(model_name:='DeepChem/ChemBERTa-77M-MLM')}")
    print(f"INFO: model_dir: {model_dir}")

    seed = 42

    # load mlb
    with open(mlb_dir / "mlb.pkl", "rb") as f:
        mlb = pkl.load(f)
    
    # tokenize data, get masks, and create dataloaders
    tokenizer = SMILESTokenizer(vocab_path="tokenizer/vocab.json", download_vocab=True)
    model, device, d_model = load_model(
        drop_rate=0.1, d_out = mlb.classes_.shape[0], model_name=model_name,
    )

    if model_dir == Path("models/ensemble_single_canon_chiral_20epoch"):
        run_names = ["exalted-sweep-1", "playful-sweep-2", "bumbling-sweep-3"]
    else:
        print(f"INFO: Running inference for the single model: {model_dir}")
        run_names = [model_dir.name]
    
    if args.extract_embeddings:
        # Take the first model from the ensemble for extraction, or use a specific single model
        run_names = [run_names[0]]

    # Process in batches
    chunk_size = args.batch_size
    for i in range(0, len(X_smiles), chunk_size):
        X_smiles_batch = X_smiles[i:i+chunk_size]
        batch_end = min(i+chunk_size, len(X_smiles))

        if args.input_csv:
            df_batch = input_df.iloc[i:batch_end].copy()
            df_batch = df_batch.drop(columns=["canon_smiles"])
        else:
            df_batch = pd.DataFrame()
            df_batch["smiles"] = X_smiles_batch
            df_batch["compound_id"] = args.compound_id or "molecule_0"

        print(f"INFO: Getting predictions for batch {i} to {batch_end}. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
        
        preds = []
        for run_name in run_names:
            # load best model
            model_path = model_dir if model_dir.name == run_name else Path(f"models/{run_name}")
            model.load_state_dict(torch.load(model_path / "best_model.pt", map_location=device), strict=False) 
            print(f"INFO: Getting probs from model {run_name}. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
            predictions = get_probs_from_model(
                model,
                X_smiles_batch,
                tokenizer,
                device,
                t_curr,
                batch_size=batch_size,
                seed=seed,
                extract_embeddings=args.extract_embeddings,
            )
            
            if args.extract_embeddings:
                preds.append(predictions) # For embeddings we just use first model
                break
            else:
                preds.append(pd.DataFrame(predictions, columns=mlb.classes_))

        if args.extract_embeddings:
            embeddings_arr = preds[0]
            
            # Save metadata (the batch dataframe) and embeddings together in a .pt dictionary
            batch_data = {
                "metadata": df_batch,
                "embeddings": torch.tensor(embeddings_arr)
            }
            
            if args.input_csv:
                save_dir = Path("inference_results", model_dir.name, "embeddings", input_name)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(batch_data, save_dir / f"{input_name}_{i}_{batch_end}_embeddings.pt")
                print(f"Saved embeddings and metadata to: {save_dir / f'{input_name}_{i}_{batch_end}_embeddings.pt'}")
            else:
                save_dir = Path("inference_results", model_dir.name, "embeddings", "single_smiles")
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(batch_data, save_dir / f"single_smiles_{i}_{batch_end}_embeddings.pt")
                print(f"Saved embeddings and metadata to: {save_dir / f'single_smiles_{i}_{batch_end}_embeddings.pt'}")
        else:
            # average predictions across models
            mean_probs = np.mean(preds, axis=0)

            if args.p_threshold is not None:
                # Filter significantly terms
                high_prob_indices = mean_probs >= args.p_threshold
                filtered_dict_list = []
                for j in range(len(high_prob_indices)):
                    filtered_probs = mean_probs[j][high_prob_indices[j]]
                    filtered_labels = mlb.classes_[high_prob_indices[j]]
                    filtered_dict = {filtered_labels[k]: filtered_probs[k] for k in range(len(filtered_probs))}
                    filtered_dict_list.append(filtered_dict)
                df_batch["top_preds"] = filtered_dict_list
            else:
                # include all probs like originally
                preds_df = pd.DataFrame(mean_probs, columns=mlb.classes_)
                df_batch = pd.concat([df_batch, preds_df], axis=1)
                
            if args.input_csv:
                save_dir = Path("inference_results", model_dir.name, "predictions", input_name)
                save_dir.mkdir(parents=True, exist_ok=True)
                df_batch.to_csv(save_dir / f"{input_name}_{i}_{batch_end}.csv", index=False)
            else:
                save_dir = Path("inference_results", model_dir.name, "predictions", "single_smiles")
                save_dir.mkdir(parents=True, exist_ok=True)
                df_batch.to_csv(save_dir / f"single_smiles_{i}_{batch_end}.csv", index=False)
                
    print(f"INFO: Total time: {round(abs((t_old:=t_start) - (t_curr:=time.time())), 3)} seconds")
    

if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", type=str, help="Single SMILES string for prediction")
    input_group.add_argument("--input_csv", type=str, help="Path to a CSV file containing SMILES strings")
    
    parser.add_argument("--compound_id", type=str, default="", help="Literal ID or name for the single molecule (if using --smiles)")
    parser.add_argument("--smiles_column", type=str, default="smiles", help="Name of the SMILES column (if using --input_csv)")
    
    parser.add_argument("--isomericSmiles", default=True, help="chiral or achiral for smiles string canonicalization")
    parser.add_argument("--mlb_dir", default="../../../data/final_datasets/preprocessed_propagated_1_hard", help="directory of mlb. This is typically where the dataset is stored")
    parser.add_argument("--model_dir", default="models/ensemble_single_canon_chiral_20epoch", help="directory of ensemble model results")
    parser.add_argument("--p_threshold", type=float, default=None, help="probability threshold for saving only significant terms. If set, predictions below this are discarded.")
    parser.add_argument("--batch_size", type=int, default=100000, help="batch size for processing huge files")
    parser.add_argument("--extract_embeddings", action="store_true", help="whether to extract embeddings instead of making predictions")
    args = parser.parse_args()
    main(args)