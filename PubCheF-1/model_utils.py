import time
import multiprocessing as mp
from pathlib import Path

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
from tokenizers import Tokenizer, Regex
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from utils import canon_smiles_single_random


class ChemMLMRegressor(nn.Module):
    """
    Classification on ChemBERTa
    """

    def __init__(
        self,
        base_model_name="DeepChem/ChemBERTa-77M-MLM",
        model_initialization="pretrained",
        drop_rate=0.2,
        d_out=1,
    ):
        super(ChemMLMRegressor, self).__init__()
        if model_initialization == "pretrained":
            self.bert = AutoModelForMaskedLM.from_pretrained(base_model_name)
        else:
            self.bert = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(base_model_name))

        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.bert.config.hidden_size, d_out),
        )
        # Set via model.extract_embeddings = True before calling forward;
        # avoids passing kwargs through DataParallel which causes scatter issues.
        self.extract_embeddings = False

    def forward(self, input_ids, attention_masks):
        """
        Called when model object is called
        """
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True
        )
        if self.extract_embeddings:
            return {"embedding": outputs["hidden_states"][-1][:, 0, :]}
        predictions = self.regressor(outputs["hidden_states"][-1][:, 0, :])  # state of cls token
        return {"predictions": predictions}


class SMILESTokenizer:
    """
    Tokenizer for SMILES strings. AutoTokenizer using DeepChem/ChemBERTa-77M-MLM was broken and this fixes it.
    Requires the vocab.json file from the DeepChem/ChemBERTa-77M-MLM model.
    """

    _default_tokenizer_dir = Path(__file__).parent / "tokenizer"

    def __init__(self, vocab_path=None, download_vocab=False):
        if vocab_path is None:
            vocab_path = self._default_tokenizer_dir / "vocab.json"
        if download_vocab:
            tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self._default_tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(self._default_tokenizer_dir))

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


def load_model(
    base_model_name="DeepChem/ChemBERTa-77M-MLM",
    model_initialization="pretrained",
    drop_rate=0.2,
    d_out=1,
):
    model = ChemMLMRegressor(
        base_model_name=base_model_name,
        model_initialization=model_initialization,
        drop_rate=drop_rate,
        d_out=d_out,
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


def create_dataloader(X, tokenizer, y=None, batch_size=32, seed=True, shuffle=True):
    """
    Create a DataLoader for the given data.

    If y is provided, labels are included in the dataset (3-tensor: inputs, masks, labels).
    If y is None, only inputs and masks are included (2-tensor), for inference without labels.
    """
    # tokenize data, get masks, and create dataloaders
    encoded_corpus = tokenizer(
        text=X.tolist(),
    )

    inputs = np.array(encoded_corpus["input_ids"])
    masks = np.array(encoded_corpus["attention_mask"])
    masks = masks.astype(bool)

    if y is not None:
        labels = y.astype(bool)
        dataset = TensorDataset(
            torch.tensor(inputs),
            torch.tensor(masks),
            torch.tensor(labels),
        )
    else:
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
        y=None,
        batch_size=256,
        seed=42,
        extract_embeddings=False,
        random_canon=False,
        ):
    """
    Get predictions/embeddings from model.

    If y is provided, returns (predictions, labels). Otherwise returns predictions only.
    If extract_embeddings is True, returns CLS token embeddings instead of classification probabilities.
    """
    if random_canon:
        with mp.Pool(mp.cpu_count()) as p:
            X_smiles = np.array(p.map(canon_smiles_single_random, X_smiles))

    X_smiles = np.array(X_smiles)
    dataloader = create_dataloader(X_smiles, tokenizer, y=y, batch_size=batch_size, seed=seed, shuffle=False)

    # Set extract_embeddings mode on the underlying module (avoids passing kwargs through DataParallel)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.extract_embeddings = extract_embeddings

    all_predictions = []
    all_labels = [] if y is not None else None
    model.eval()
    with torch.no_grad():
        for count, batch in enumerate(dataloader):
            if count % 500 == 0:
                print(f"INFO: Batch {count} of {len(dataloader)}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds")
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            outputs = model(b_input_ids, b_input_mask)
            if extract_embeddings:
                predictions = outputs["embedding"].cpu().numpy()
            else:
                predictions = outputs["predictions"].cpu().numpy()
                # convert logits to probabilities
                predictions = 1 / (1 + np.exp(-predictions))

            all_predictions.append(predictions)

            if y is not None:
                b_labels = batch[2].float().cpu().numpy()
                all_labels.append(b_labels)

    all_predictions = np.concatenate(all_predictions)

    if y is not None:
        all_labels = np.concatenate(all_labels)
        return all_predictions, all_labels
    return all_predictions


def eval_model(model,
               X_eval,
               y_eval,
               tokenizer,
               loss_function,
               device,
               save_path,
               mlb,
               n_labels,
               t_curr=None,
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
    if t_curr is None:
        t_curr = time.time()

    if random_canon:
        with mp.Pool(mp.cpu_count()) as p:
            X_eval = np.array(p.map(canon_smiles_single_random, X_eval))
    eval_dataloader = create_dataloader(
        X_eval, tokenizer, y=y_eval, batch_size=batch_size, seed=seed, shuffle=False,
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
        import wandb
        wandb.log({f"{split_type}_mean_loss": mean_eval_loss, f"{split_type}_mean_roc_auc": mean_roc_auc, f"{split_type}_mean_pr_auc": mean_pr_auc, f"{split_type}_mean_brier_score": mean_brier_score})

    # save mean ROC-AUC and PR-AUC
    with open(save_path / f"{split_type}_metrics_mean.txt", "w") as f:
        f.write(f"Mean Loss: {mean_eval_loss:.6f}\n")
        f.write(f"Mean ROC-AUC: {mean_roc_auc:.6f}\n")
        f.write(f"Mean PR-AUC: {mean_pr_auc:.6f}\n")
        f.write(f"Mean Brier Score: {mean_brier_score:.6f}\n")
