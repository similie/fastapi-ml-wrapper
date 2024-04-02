import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from os import path, getcwd
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
from mutils import collate_batch, generate_datetime_index
from preprocessor import load_data_csv
from pytorch_lightning.utilities import CombinedLoader
import pytorch_lightning as pl

# transforms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

class SequenceDataset(Dataset):
    def __init__(self, 
                 dataframe,
                 station_name, 
                 features,
                 target=['precipitation'], 
                 prediction_window=12,
                 sequence_length=12):
        self.features = features
        self.target = target
        self.station_name = station_name
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.dataframe = dataframe 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx): 
        if (idx+self.sequence_length) > len(self.dataframe):
            indexes = list(range(idx, len(self.dataframe)))
        else:
            indexes = list(range(idx, idx + self.sequence_length))
        X = self.dataframe[self.features].iloc[indexes, :].values
        Y = self.dataframe[self.target].shift(
            periods=self.prediction_window, freq='h'
        ).iloc[indexes].values
        return torch.tensor(X).float(), torch.tensor(Y).float()
