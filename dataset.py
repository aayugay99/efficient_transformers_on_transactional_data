import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def transaction_collate_fn(batch, pad_to_size=100):
    batch_dict = {}
    for key in batch[0].keys():
        list_of_sequences = [torch.LongTensor(d[key]) for d in batch]
        list_of_sequences.append(torch.zeros((pad_to_size),dtype=int))
        padded_sequences = pad_sequence(list_of_sequences, batch_first=True, padding_value=0)
        padded_sequences = padded_sequences[:-1]
        batch_dict[key] = padded_sequences

    return batch_dict


class TransactionDataset(Dataset):
    def __init__(self, df, id_col=None, dt_col=None, cat_cols=[], min_length=0, max_length=1000, random_slice=True):
        super().__init__()

        df = df.sort_values(dt_col)
        df_agg = df.groupby(id_col).agg({col: lambda x: x.tolist() for col in cat_cols})

        length = df_agg[cat_cols[0]].apply(lambda x: len(x))
        filter = (length >= min_length)
        df_filtered = df_agg[filter]
        if len(df_agg) - len(df_filtered) > 0:
            print(f"{len(df_agg) - len(df_filtered)} sequences were filtered")

        self.min_length = min_length
        self.max_length = max_length
        self.random_slice = random_slice
        self.cat_cols = cat_cols
        self.sequences = df_filtered.to_dict(orient="list")

    def __getitem__(self, idx):
        seq_len = len(self.sequences[self.cat_cols[0]][idx])
        start = 0
        if self.random_slice:
            start = np.random.randint(0, seq_len - self.min_length + 1)
        return {k: v[idx][start: start + self.max_length] for k, v in self.sequences.items()}
    
    def __len__(self):
        return len(self.sequences[self.cat_cols[0]])
