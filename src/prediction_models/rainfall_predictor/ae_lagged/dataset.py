
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, IterableDataset
import pandas as pd
import numpy as np
from math import ceil
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
from preprocessor import load_data_csv, load_data_json
from pytorch_lightning.utilities import CombinedLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule

class LaggedData(LightningDataModule):
    def __init__(self, data_path: str,
                 target: str = None, 
                 batch_size: int = 32, 
                 loader_fn = load_data_csv,):
        self.loader_fn = loader_fn
        self.target = target
        self.batch_size = batch_size
        self.df, self.transforms, self.features = self.loader_fn(data_path, pred=None)
        self.stations = self.df.station.value_counts().index.to_list()

    def setup(self, station_id: str, stage: str):
        if stage == None:
            _df = self.df[self.df.station == station_id].drop('station', axis=1)
            self.train_set = LaggedDataset(_df.iloc[0:ceil(len(_df)*.6)])
            self.val_set = LaggedDataset(_df.iloc[ceil(len(_df)*.6):ceil(len(_df)*.8)])
            self.test_set = LaggedDataset(_df.iloc[ceil(len(_df)*.8):])
        if stage == "pred":
            self.predict_set = LaggedDataset(_df)
        
    def train_dataloader(self):
        return DataLoader(self.train_set, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          collate_fn=pad_collate,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          collate_fn=pad_collate,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          collate_fn=pad_collate,
                          drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          collate_fn=pad_collate,
                          drop_last=True)
    
    
class LaggedDataset(Dataset):
    def __init__(self, df, sequence_length: int = 12, target: str = "precipitation"):
        self.df = df
        self.sequence_length = sequence_length
        self.target = target
        self.features = self.df.drop(self.target, axis=1).columns.to_list()
        X = self.df[self.features]
        Y = self.df[self.target].shift(     
            periods=-1, freq='h'
        )

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if (idx+self.sequence_length) > len(self.df):
            x, y = self.pad_df(idx)
        else:
            indexes = list(range(idx, idx + self.sequence_length))
            x = self.X.iloc[indexes, :].values
            y = self.Y.iloc[indexes, :].values
        return torch.tensor(x).float(), torch.tensor(y).float()

    def pad_df(self, idx):
            gap = idx+self.sequence_length - len(self.df)             
            indexes = list(range(idx, len(self.df)))
            x_to_pad = self.X.iloc[indexes, :]
            y_to_pad = self.Y.iloc[indexes, :]
            xpad = pd.concat([x_to_pad.iloc[:,0]]*gap).reset_index(drop=True)
            ypad = pd.concat([y_to_pad.iloc[:,0]]*gap).reset_index(drop=True)
            x = pd.concat([x_to_pad, xpad]).reset_index(drop=True).values
            y = pd.concat([y_to_pad, ypad]).reset_index(drop=True).values
        return x, y
        
def pad_collate(batch):
    x = [item[0] for item in batch]
    x = pad_sequence(x, batch_first=True)
    y = [item[1] for item in batch]
    y = pad_sequence(y, batch_first=True)
    return x, y
