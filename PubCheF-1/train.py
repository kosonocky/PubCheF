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
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast

from model_utils import (
    ChemMLMRegressor,
    SMILESTokenizer,
    FocalLoss,
    NoamLR,
    load_model,
    create_dataloader,
    eval_model,
)
from utils import (
    canon_smiles,
    canon_smiles_single_random,
    smooth_labels,
    df_to_x_y,
    shuffle_tokens,
    save_plot_losses,
)


# keeping off because it slows down training
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

os.environ['RDMAV_HUGEPAGES_SAFE'] = '1' # this is needed to avoid a warning from transformers
os.environ['NCCL_DEBUG'] = 'INFO' # this is needed to debug distributed training issues, but it slows down training so keeping it off for now


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
            X_train, tokenizer, y=y_train, batch_size=batch_size, seed=seed, shuffle=True,
        )
        validation_dataloader = create_dataloader(
            X_val, tokenizer, y=y_val, batch_size=batch_size, seed=seed, shuffle=False,
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
            import wandb
            wandb.log({"validation_loss": validation_loss, "train_loss": train_loss, "epoch": epoch})
        print(
            f"INFO: Train loss: {train_loss}, Validation loss: {validation_loss}, Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds"
        )

        # save model to checkpoints folder
        Path(save_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "checkpoints" / f"model_e{epoch}.pt")

    return train_losses, validation_losses


def main(args):
    t_start = time.time()
    t_curr = t_start

    use_wandb = args.wandb
    debug = args.debug

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

    if debug:
        train_df = pd.read_csv(Path(data_dir, split_type, "train.csv"), nrows=1000)
        val_df = pd.read_csv(Path(data_dir, split_type, "val.csv"), nrows=1000)
        test_df = pd.read_csv(Path(data_dir, split_type, "test.csv"), nrows=1000)
    else:
        train_df = pd.read_csv(Path(data_dir, split_type, "train.csv"))
        val_df = pd.read_csv(Path(data_dir, split_type, "val.csv"))
        test_df = pd.read_csv(Path(data_dir, split_type, "test.csv"))

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
        drop_rate=drop_rate, d_out=y_train.shape[1], base_model_name=model_name, model_initialization=model_initialization,
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
        import wandb
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
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode with smaller dataset and fewer epochs for quick testing")
    args = parser.parse_args()
    main(args)
