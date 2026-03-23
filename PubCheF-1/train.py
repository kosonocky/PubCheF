import os
import time
import random
import argparse
import pickle as pkl
from pathlib import Path
from ast import literal_eval
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
from rdkit import Chem


# keeping off because it slows down training
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

os.environ['RDMAV_HUGEPAGES_SAFE'] = '1' # this is needed to avoid a warning from transformers
os.environ['NCCL_DEBUG'] = 'INFO' # this is needed to debug distributed training issues, but it slows down training so keeping it off for now

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

    def forward(self, input_ids, attention_masks):
        """
        Called when model object is called
        """
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True
        )
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
    Noam learning rate scheduler from Attention is All You Need. This is a custom scheduler that is not available in PyTorch.
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
    
def canon_smiles(smiles, isomericSmiles=True, kekuleSmiles=False):
    try: 
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles)
    except Exception as e:
        print(e)
        print(f"ERROR: {smiles} set to None")
        return None

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
    
def shuffle_tokens(string, tokenizer):
    """
    Shuffle string by tokenizing, shuffling, and decoding back into string
    """
    ids = tokenizer.tokenizer.encode(string).ids
    random.shuffle(ids)
    return "".join(tokenizer.tokenizer.decode(ids).split())

def smooth_labels(y, alpha=0.1):
    """
    Label smoothing
    """
    with torch.no_grad():
        y = y * (1 - alpha) + (alpha / y.shape[1])
    return y

def df_to_x_y(df, mlb):
    y = mlb.transform(df['labels'])
    print(f"INFO: y.shape: {y.shape}")
    X = df['smiles'].values
    cid = df["cid"].values
    return X, y, cid

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

        # NOTE: Turning this off for now as it's causing issues
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)
    return model, device, d_model


def create_dataloader(X, y, tokenizer, batch_size=32, seed=True, shuffle=True):

    # tokenize data, get masks, and create dataloaders
    encoded_corpus = tokenizer(
        text=X.tolist(),
    )

    inputs = np.array(encoded_corpus["input_ids"])
    masks = np.array(encoded_corpus["attention_mask"])
    masks = masks.astype(bool)
    labels = y.astype(bool)

    # create dataloaders
    dataset = TensorDataset(
        torch.tensor(inputs),
        torch.tensor(masks),
        torch.tensor(labels),
    )

    generator = torch.Generator().manual_seed(seed)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
    )

    return dataloader

def train_model(
    model,
    tokenizer,
    X_train,
    y_train,
    X_val,
    y_val,
    optimizer,
    scheduler,
    loss_function,
    device,
    save_path,
    start_epoch=0,
    epochs=100,
    random_canon=False,
    batch_size=256,
    seed=42,
    t_curr=time.time(),
    use_wandb=False,
):    
    # train model (this is a multilabel classification task)
    train_losses = []
    validation_losses = []

    scaler = GradScaler() # mixed precision training

    for epoch in range(start_epoch, epochs):
        print(f"INFO: Epoch {epoch}/{epochs}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

        if random_canon:
            with mp.Pool(mp.cpu_count()) as p:
                X_train = np.array(p.map(canon_smiles_single_random, X_train))
                X_val = np.array(p.map(canon_smiles_single_random, X_val))
        train_dataloader = create_dataloader(
            X_train, y_train, tokenizer, batch_size=batch_size, seed=seed, shuffle=True,
        )
        validation_dataloader = create_dataloader(
            X_val, y_val, tokenizer, batch_size=batch_size, seed=seed, shuffle=False,
        )

        # training loop
        model.train()
        train_loss = 0
        for count, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].float().to(device)

            model.zero_grad()
            
            with autocast():
                outputs = model(b_input_ids, b_input_mask)
                predictions = outputs["predictions"]
                loss = loss_function(predictions, b_labels)

            train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if count % 500 == 0:
                print(f"INFO: Batch {count}/{len(train_dataloader)}. Batch loss {loss.item()}", end="\r")
                with open(save_path / "batch_losses.csv", "a") as f:
                    f.write(f"{epoch},{count},{loss.item()}\n")

        train_loss = train_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # validation loop
        validation_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].float().to(device)

                outputs = model(b_input_ids, b_input_mask)
                predictions = outputs["predictions"]

                loss = loss_function(predictions, b_labels)
                validation_loss += loss.item()

        validation_loss = validation_loss / len(validation_dataloader)
        validation_losses.append(validation_loss)

        if use_wandb:
            wandb.log({"validation_loss": validation_loss, "train_loss": train_loss, "epoch": epoch})
        print(
            f"INFO: Train loss: {train_loss}, Validation loss: {validation_loss}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds"
        )

        # save model to checkpoints folder
        Path(save_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "checkpoints" / f"model_e{epoch}.pt")

    return train_losses, validation_losses

def save_plot_losses(train_losses, validation_losses, save_path):
    np.save(save_path / "train_losses.npy", np.array(train_losses))
    np.save(save_path / "validation_losses.npy", np.array(validation_losses))
    pd.DataFrame(
        {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
    ).to_csv(save_path / "train_val_losses.csv", index=False)

    # plot losses with log scale y axis
    plt.plot(train_losses, label="Train loss")
    plt.plot(validation_losses, label="Validation loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path / "losses.png")
    plt.close()

def eval_model(model, 
               X_eval,
               y_eval, 
               tokenizer,
               loss_function, 
               device, 
               save_path, 
               mlb, 
               n_labels, 
               t_curr=time.time(),
               random_canon=False,
               batch_size=256,
               seed=42,
               use_wandb=False,
               split_type="test", # test, val, train, if you want to calc these metrics for each split
               ):
    """
    Eval model. Track ROC-AUC, PR-AUC, and Brier Score for each label and average across all labels (as well as loss)

    This was originally made just for the test set but was modified to be able to calculate these metrics for the train and val sets as well
    """

    if random_canon:
        with mp.Pool(mp.cpu_count()) as p:
            X_eval = np.array(p.map(canon_smiles_single_random, X_eval))
    eval_dataloader = create_dataloader(
        X_eval, y_eval, tokenizer, batch_size=batch_size, seed=seed, shuffle=False,
    )

    eval_loss = 0
    eval_predictions = []
    eval_labels = []
    model.eval()
    with torch.no_grad():
        for count, batch in enumerate(eval_dataloader):
            if count % 500 == 0:
                print(f"INFO: Batch {count} of {len(eval_dataloader)}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].float().to(device)

            outputs = model(b_input_ids, b_input_mask)
            predictions = outputs["predictions"]

            loss = loss_function(predictions, b_labels)
            eval_loss += loss.item()

            # convert predictions and labels to numpy arrays
            predictions = predictions.cpu().numpy()
            b_labels = b_labels.cpu().numpy()
            
            # convert logits to probabilities
            predictions = 1 / (1 + np.exp(-predictions))
            
            eval_predictions.append(predictions)
            eval_labels.append(b_labels)

    # calculate ROC-AUC and PR-AUC for each label
    # ignore labels that are all 0s or 1s
    eval_predictions = np.concatenate(eval_predictions)
    eval_labels = np.concatenate(eval_labels)
    roc_auc = []
    pr_auc = []
    brier_scores = []
    for i in range(n_labels):
        try:
            roc_auc.append(roc_auc_score(eval_labels[:, i], eval_predictions[:, i]))
            pr_auc.append(average_precision_score(eval_labels[:, i], eval_predictions[:, i]))
            brier_scores.append(brier_score_loss(eval_labels[:, i], eval_predictions[:, i]))
        except ValueError as e:
            roc_auc.append(0)
            pr_auc.append(0)
            brier_scores.append(brier_score_loss(eval_labels[:, i], eval_predictions[:, i]))

    print(f"Len of roc_auc: {len(roc_auc)}")
    print(f"Len of pr_auc: {len(pr_auc)}")
    print(f"Len of brier_scores: {len(brier_scores)}")
    print(f"Len of labels: {len(mlb.classes_)}")
    # save df of roc_auc, pr_auc, and labels
    df = pd.DataFrame(
        {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier_scores,
            "labels": mlb.classes_,
        }
    )
    df.to_csv(save_path / f"{split_type}_metrics_indiv.csv", index=False)

    mean_eval_loss = eval_loss / len(eval_dataloader)

    # ignore zero values as these are labels not found in the eval set
    mean_roc_auc = np.mean([roc for roc in roc_auc if roc != 0])
    mean_pr_auc = np.mean([pr for pr in pr_auc if pr != 0])
    mean_brier_score = np.mean(brier_scores) # this is fine to leave in as brier scores are always calculated

    if use_wandb:
        wandb.log({f"{split_type}_mean_loss": mean_eval_loss, f"{split_type}_mean_roc_auc": mean_roc_auc, f"{split_type}_mean_pr_auc": mean_pr_auc, f"{split_type}_mean_brier_score": mean_brier_score})

    # save mean ROC-AUC and PR-AUC
    with open(save_path / f"{split_type}_metrics_mean.txt", "w") as f:
        f.write(f"Mean Loss: {mean_eval_loss:.6f}\n")
        f.write(f"Mean ROC-AUC: {mean_roc_auc:.6f}\n")
        f.write(f"Mean PR-AUC: {mean_pr_auc:.6f}\n")
        f.write(f"Mean Brier Score: {mean_brier_score:.6f}\n")


def main(args):
    t_start = time.time()
    t_curr = t_start

    use_wandb = args.wandb
    
    if use_wandb:
        import wandb
        print("Using wandb")
        # sweep from yaml
        wandb.init(project="PubCheF-1")
        config = wandb.config

        # set save path
        save_path = Path(f"models/{wandb.run.name}")
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Saving to {save_path}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

        # save wandb config args
        print("Configuration details:")
        with open(save_path / "config.txt", "w") as f:
            f.write(f"Configuration details:\n")
            for arg, value in vars(config).items():
                f.write(f"{arg}: {value}\n")
                print(f"{arg}: {value}")
            f.write(f"gpus available: {torch.cuda.device_count()}\n")
            print(f"gpus available: {torch.cuda.device_count()}")

        #unpack args
        split_type = config.split_type
        drop_rate = config.drop_rate
        epochs = config.epochs
        batch_size = config.batch_size
        opt_lr = config.opt_lr
        opt_eps = config.opt_eps
        opt_weight_decay = config.opt_weight_decay
        num_warmup_steps = config.num_warmup_steps
        focal_loss = config.focal_loss
        model_name = config.model_name
        model_initialization = config.model_initialization
        data_dir = config.data_dir
        seed = config.seed
        label_smoothing_alpha = config.label_smoothing_alpha
        random_canon = config.random_canon
        pos_class_weight = config.pos_class_weight
        isomericSmiles = config.isomericSmiles
        kekuleSmiles = config.kekuleSmiles
        drop_complexes = False # currently not a hyperparameter we're sweeping for
        shuffle_smiles = config.shuffle_smiles
        smiles_elements_only = config.smiles_elements_only

    else:
        print("Not using wandb. Using args defined below")
        # NOTE For testing / non-sweep training
        split_type = "scaffold_split"
        drop_rate = 0.1
        epochs = 20
        batch_size = 128
        opt_lr = 1e-3
        opt_eps = 1e-8
        opt_weight_decay = 1e-2
        num_warmup_steps = 2000
        focal_loss = True # use focal loss if True, otherwise use BCEWithLogitsLoss with pos_weight if pos_class_weight > 0, otherwise just BCEWithLogitsLoss
        model_name = "DeepChem/ChemBERTa-77M-MLM"
        model_initialization = "pretrained"
        data_dir = "../data/final_datasets/preprocessed_propagated_1_hard"
        seed = 42
        label_smoothing_alpha = 0
        random_canon = False # default false means do not randomly canonicalize each epoch. This is a form of data augmentation that we thought would help with generalization, but it showed little effect
        pos_class_weight = 0 # zero means don't balance classes, which is fine if using focal loss with alpha
        isomericSmiles = True # default True; use chirality
        kekuleSmiles = False # default false
        drop_complexes = False # currently not a hyperparameter we're sweeping for
        shuffle_smiles = False # default false; only used in ablation, randomly shuffle tokens in the SMILES string as a form of data augmentation to test if model is learning meaningful representations of the SMILES strings or just memorizing them
        smiles_elements_only = False # default false; only keep element symbols in the SMILES strings, removing all connectivity information, as a form of ablation to test if model is learning meaningful representations of the SMILES strings or just memorizing them
        # create arbitrary save path
        save_path = Path(f"models/test")
        save_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # train_df = pd.read_csv(Path(data_dir, split_type, "train.csv"))
    # val_df = pd.read_csv(Path(data_dir, split_type, "val.csv"))
    # test_df = pd.read_csv(Path(data_dir, split_type, "test.csv"))

    # NOTE for testing
    train_df = pd.read_csv(Path(data_dir, split_type, "train.csv"), nrows=1000)
    val_df = pd.read_csv(Path(data_dir, split_type, "val.csv"), nrows=1000)
    test_df = pd.read_csv(Path(data_dir, split_type, "test.csv"), nrows=1000)
    
    train_df['labels'] = train_df['labels'].apply(literal_eval)
    val_df['labels'] = val_df['labels'].apply(literal_eval)
    test_df['labels'] = test_df['labels'].apply(literal_eval)

    # canonicalize smiles strings if needed
    if not ((isomericSmiles is True) and (kekuleSmiles is False)):
        print(f"INFO: Canonicalizing smiles strings based on isomericSmiles: {isomericSmiles}, kekuleSmiles: {kekuleSmiles}. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
                
        train_df['smiles'] = train_df['smiles'].apply(lambda x: canon_smiles(x, isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles))
        train_df = train_df.dropna(subset=['smiles'])
        val_df['smiles'] = val_df['smiles'].apply(lambda x: canon_smiles(x, isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles))
        val_df = val_df.dropna(subset=['smiles'])
        test_df['smiles'] = test_df['smiles'].apply(lambda x: canon_smiles(x, isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles))
        test_df = test_df.dropna(subset=['smiles'])

    if drop_complexes:
        def merge_sets(series):
            result = set()
            for d in series:
                result = result.union(d)
            return list(result)
        print(f"INFO: Dropping complexes. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
        train_df['smiles'] = train_df['smiles'].apply(lambda x: x.split(".")[0])
        train_df = train_df.groupby('smiles').agg({'labels': merge_sets, 'cid': 'first'}).reset_index()
        val_df['smiles'] = val_df['smiles'].apply(lambda x: x.split(".")[0])
        val_df = val_df.groupby('smiles').agg({'labels': merge_sets, 'cid': 'first'}).reset_index()
        test_df['smiles'] = test_df['smiles'].apply(lambda x: x.split(".")[0])
        test_df = test_df.groupby('smiles').agg({'labels': merge_sets, 'cid': 'first'}).reset_index()

    if smiles_elements_only:
        print(f"INFO: Converting smiles strings to elements only. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
        train_df['smiles'] = train_df['smiles'].apply(lambda x: "".join([i for i in x if i.isalpha()]))
        val_df['smiles'] = val_df['smiles'].apply(lambda x: "".join([i for i in x if i.isalpha()]))
        test_df['smiles'] = test_df['smiles'].apply(lambda x: "".join([i for i in x if i.isalpha()]))

    # load custom working tokenizer. Necessary to use this since the huggingface one was broken
    tokenizer = SMILESTokenizer(vocab_path="tokenizer/vocab.json", download_vocab=True)

    if shuffle_smiles:
        print(f"INFO: Shuffling smiles strings. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
        train_df['smiles'] = train_df['smiles'].apply(lambda x: shuffle_tokens(x, tokenizer))
        val_df['smiles'] = val_df['smiles'].apply(lambda x: shuffle_tokens(x, tokenizer))
        test_df['smiles'] = test_df['smiles'].apply(lambda x: shuffle_tokens(x, tokenizer))

    with open(Path(data_dir, "mlb.pkl"), "rb") as f:
        mlb = pkl.load(f)
    
    # convert df to X, y, and cid
    X_train, y_train, cid_train = df_to_x_y(train_df, mlb)
    X_val, y_val, cid_val = df_to_x_y(val_df, mlb)
    X_test, y_test, cid_test = df_to_x_y(test_df, mlb)
    print(f"INFO: Data loaded, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # choose loss function. focal loss implemented as most labels are zero
    if pos_class_weight > 0:
        if focal_loss:
            loss_function = FocalLoss(alpha=pos_class_weight)
            print(f"INFO: FocalLoss alpha: {pos_class_weight}")
        else:
            loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_class_weight) # multilabel cross entropy loss
            print(f"INFO: BCEwithLogitsLoss pos_weight: {pos_class_weight}")
    else:
        if focal_loss:
            loss_function = FocalLoss(alpha=None)
            print(f"INFO: FocalLoss alpha: None")
        else:
            loss_function = nn.BCEWithLogitsLoss()
            print(f"INFO: BCEwithLogitsLoss pos_weight: None")

    # label smoothing if desired
    if label_smoothing_alpha > 0:
        y_train = smooth_labels(y_train, alpha=label_smoothing_alpha) # smooth labels
        print(f"INFO: Train labels smoothed, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    

    model, device, d_model = load_model(
        drop_rate=drop_rate, d_out = y_train.shape[1], model_name=model_name, model_initialization=model_initialization,
    )
    optimizer = AdamW(params=model.parameters(), lr=opt_lr, eps=opt_eps, weight_decay=opt_weight_decay)
    scheduler = NoamLR(optimizer=optimizer, d_model=d_model, warmup_steps=num_warmup_steps)

    # train model
    train_losses, validation_losses = train_model(
        model=model,
        tokenizer=tokenizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        device=device,
        save_path=save_path,
        start_epoch=0,
        epochs=epochs,
        random_canon=random_canon,
        batch_size=batch_size,
        seed=seed,
        use_wandb=use_wandb,
    )
    print(f"INFO: Model trained, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # save and plot losses
    save_plot_losses(train_losses, validation_losses, save_path)
    print(f"INFO: Losses saved and plotted, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    best_epoch = np.argmin(validation_losses)
    print(f"INFO: Best epoch: {best_epoch}")
    model.load_state_dict(torch.load(save_path / "checkpoints" / f"model_e{best_epoch}.pt")) # load best model
    torch.save(model.state_dict(), save_path / "best_model.pt") # save best model under new name
    # remove all checkpoints that are not the best epoch or every 5 epochs
    for file in (save_path / "checkpoints").iterdir():
        if (int(file.stem.split("_e")[1]) + 1) % 5 != 0:
            if int(file.stem.split("_e")[1]) != 0:
                file.unlink()
    if use_wandb:
        wandb.log({"best_epoch": best_epoch}) # log best epoch
        artifact = wandb.Artifact("best_model", type="model") # create artifact
        artifact.add_file(save_path / "best_model.pt") # save best model to wandb
        wandb.log_artifact(artifact) # log artifact
        print(f"INFO: Saved best epoch to local & wandb. Removed all other checkpoints. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    
    print(f"INFO: Evaluating best model on validation set. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
    eval_model(
        model=model,
        X_eval=X_val,
        y_eval=y_val,
        tokenizer=tokenizer,
        loss_function=loss_function,
        device=device,
        save_path=save_path,
        mlb=mlb,
        n_labels=y_val.shape[1],
        t_curr=t_curr,
        random_canon=random_canon,
        batch_size=batch_size,
        seed=seed,
        use_wandb=use_wandb,
        split_type="val",
    )

    print(f"INFO: Evaluating best model on test set. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")

    # test model from best epoch
    eval_model(
        model=model,
        tokenizer=tokenizer,
        X_eval=X_test,
        y_eval=y_test,
        loss_function=loss_function,
        device=device,
        save_path=save_path,
        mlb=mlb,
        n_labels=y_test.shape[1],
        t_curr=t_curr,
        random_canon=random_canon,
        batch_size=batch_size,
        seed=seed,
        use_wandb=use_wandb,
        split_type="test",
    )
    print(f"INFO: Total time: {round(abs((t_old:=t_start) - (t_curr:=time.time())), 3)} seconds")
    

if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser(description="Train ChemMLMRegressor on PubCheF-1 dataset")
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb for logging and hyperparameter sweeps")
    args = parser.parse_args()
    main(args)