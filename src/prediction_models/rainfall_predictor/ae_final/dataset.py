import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from os import path, getcwd
from typing import Callable
from mutils import collate_batch, generate_datetime_index
from preprocessor import load_data_csv, load_data_json
from pytorch_lightning.utilities import CombinedLoader

from preprocessor import transforms

class SequenceDataset(Dataset):
    def __init__(self, 
                 dataframe,
                 station_name, 
                 features,
                 target=None, 
                 prediction_window=12,
                 sequence_length=12):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.dataframe = dataframe 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.target == None:
            self.target = self.features
        if (idx+self.sequence_length) > len(self.dataframe):
            indexes = list(range(idx, len(self.dataframe)))
        else:
            indexes = list(range(idx, idx + self.sequence_length))
        X = self.dataframe[self.features].iloc[indexes, :].values
        if self.features == self.target:
            Y = self.dataframe[self.target].shift(
                periods=self.prediction_window, 
                freq='h').iloc[indexes, :].values
        else:
            Y = self.dataframe[self.target].iloc[indexes, :].values

        return torch.tensor(X).float(), torch.tensor(Y).float()
    
class get_dm():
    def __init__(self,
                 data_dir: str = path.join(getcwd(), '../tabula_rasa/data/combined.csv'),
                 batch_size: int = 1,
                 frames: dict | None = None,
                 features: list = [],
                 transforms: list = [],
                 target: list | None = None,
                 load_fn=load_data_csv):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target = target
        self.frames = frames
        self.load_fn = load_fn
        if self.frames == None:
            self.frames, self.transforms, self.features = load_fn(self.data_dir, pred=True) # toggle for predictions or training
        else:
            self.frames = frames
        self.sequence_datasets = self.gen_sequence_datasets(self.frames)

    def process_preds(self, plist: list):
        plist = [l[-12:][0].squeeze(0) for l in plist]
        indexes = [generate_datetime_index(v.index.max(), periods=l.size(0)) for l, v in zip(plist, self.frames.values())]
        plist = [pd.DataFrame(self.transforms[0].inverse_transform(p.numpy()), index=i, columns=self.features) for i, p in zip(indexes, plist)]
        plist = [pd.DataFrame(np.hstack((self.transforms[1].inverse_transform(p.values[:,0].reshape(-1,1)), p.values[:,1:])), index=i, columns=self.features) for i, p in zip(indexes, plist)]
        stations = list(self.frames.keys())
        preds = {}
        for s, p in zip(stations, plist):
            preds[s] = p
        return preds
    
    def gen_sequence_datasets(self, frames: dict) -> dict[str, SequenceDataset]:
        sequence_datasets = {}
        for s, _df in frames.items():
            sequence_datasets[s] = SequenceDataset(_df,
                                                   s,
                                                   features=self.features,
                                                   target=self.target)
        return sequence_datasets
             
    def gen_train_sets(self, dataset: SequenceDataset):
        split = int(len(dataset)*0.6)
        indices = np.arange(split)
        train_set = Subset(dataset, indices)
        return train_set
    
    def gen_val_loaders(self, dataset: SequenceDataset):
        split = int(len(dataset)*0.6)
        split1 = int(len(dataset)*0.8)
        indices1 = np.arange(split, split1)
        val_loader = DataLoader(Subset(dataset, indices1),
                                batch_size=self.batch_size,
                                drop_last=True,
                                shuffle=False, 
                                collate_fn=collate_batch,
                                num_workers=2)
        return val_loader

    def gen_test_loader(self, dataset: SequenceDataset):
        split = int(len(dataset)*0.8)
        indices = np.arange(split, len(dataset))
        test_loader = DataLoader(Subset(dataset, indices),
                                 batch_size=self.batch_size,
                                 drop_last=True,
                                 shuffle=False,
                                 collate_fn=collate_batch,
                                 num_workers=2)
        return test_loader

    def predict_combined_loader(self, preds: dict | None = None):
        """
            Either generates a prediction step dataloader
            from the sequence datasets of the get_dm class
            or from preds, a dictionary with station ids and 
            dataframes with weather data.
        """
        pred_loaders = {}
        if preds:
            datasets = self.gen_sequence_datasets(preds)
        else:
            datasets = self.sequence_datasets
        for s, _df in datasets.items():
            pred_loaders[s] = _df
        return CombinedLoader(pred_loaders, 'sequential') # try max_size_cycle
        
    def train_combined_loader(self):
        train_sets = []
        for _, _df in self.sequence_datasets.items():
            train_sets.append(self.gen_train_sets(_df))
        return DataLoader(ConcatDataset(train_sets),
                            batch_size=self.batch_size,
                            shuffle=False,
                            collate_fn=collate_batch,
                            num_workers=2)

    def val_combined_loader(self):
        val_loaders = {}
        for s, _df in self.sequence_datasets.items():
            val_loaders[s] = self.gen_val_loaders(_df)
        return CombinedLoader(val_loaders, 'sequential')
    
    def test_combined_loader(self):
        test_loaders = {}
        for s, _df in self.sequence_datasets.items():
            test_loaders[s] = self.gen_test_loader(_df)
        return CombinedLoader(test_loaders, 'sequential')
