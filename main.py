import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time
import wandb
from yaml import load, CLoader as Loader

from dataset import TransactionDataset, transaction_collate_fn
from utils import train_model
from models import TransformerModel

import argparse


def main(path_to_config):
    with open(path_to_config, "r") as f:
        config = load(f, Loader)

    wandb.login()

    wandb.init(
        entity="aayugay99",
        project="deep-learning-project",
        name=config["experiment_name"],
        config=config
    )

    if config["dataset"] == "rosbank":
        df = pd.read_csv('data/rosbank/train.csv')
        df['TRDATETIME'] = pd.to_datetime(df['TRDATETIME'], format='%d%b%y:%H:%M:%S')
        df = df.rename(columns={'cl_id':'client_id', 'MCC':'small_group', 'amount':'amount_rur'})
        
        mcc_to_id = {mcc: i+1 for i, mcc in enumerate(df['small_group'].unique())}

        df['amount_rur_bin'] = 1 + KBinsDiscretizer(10, encode='ordinal', subsample=None).fit_transform(df[['amount_rur']]).astype('int')
        df['small_group'] = df['small_group'].map(mcc_to_id)

    else:
        # TODO: add Sber dataset preprocessing
        pass

    clients_train, clients_val_test = train_test_split(df["client_id"].unique(), test_size=0.2, random_state=42)
    clients_val, clients_test = train_test_split(clients_val_test, test_size=0.5, random_state=42)

    train_ds = TransactionDataset(
        df[lambda x: x["client_id"].isin(clients_train)], 
        id_col="client_id", 
        dt_col="TRDATETIME", 
        cat_cols=["small_group", "amount_rur_bin"],
        min_length=config["min_length"],
        max_length=config["max_length"],
        random_slice=True
    )

    val_ds = TransactionDataset(
        df[lambda x: x["client_id"].isin(clients_val)], 
        id_col="client_id", 
        dt_col="TRDATETIME", 
        cat_cols=["small_group", "amount_rur_bin"],
        min_length=config["min_length"],
        max_length=config["max_length"],
        random_slice=False
    )

    test_ds = TransactionDataset(
        df[lambda x: x["client_id"].isin(clients_test)], 
        id_col="client_id", 
        dt_col="TRDATETIME", 
        cat_cols=["small_group", "amount_rur_bin"],
        min_length=config["min_length"],
        max_length=config["max_length"],
        random_slice=False
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=transaction_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=transaction_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=transaction_collate_fn)

    # TODO: add support for different transformers
    assert config["type"] in ["transformer", "performer", "reformer", "linear_transformer"]

    if config["type"] == "transformer":
        model = TransformerModel(**config["transformer_params"], max_len=config["max_length"])
    elif config["type"] == "performer":
        pass
    elif config["type"] == "reformer":
        pass
    elif config["type"] == "linear_transformer":
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_model(
        model, 
        optimizer, 
        {"train": train_loader, "val": val_loader, "test": test_loader}, 
        n_epochs=config["n_epochs"],
        warmup=config["warmup"],
        device=config["device"],
        save_path=config["save_path"]
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs experiments with transformers on transactional data',
    )

    parser.add_argument('filename')
    args = parser.parse_args()
    
    main(args.filename)
