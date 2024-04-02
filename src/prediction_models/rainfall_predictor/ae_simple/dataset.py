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

# transforms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

_processor = make_pipeline(StandardScaler())
# TODO use scipy.stats.skewnorm on precipitation data
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
    
class get_dm():
    def __init__(self,
                 data_dir: str = path.join(getcwd(), '../tabula_rasa/data/combined.csv'),
                 batch_size: int = 1,
                 frames = None, #dictionary
                 features: list = [],
                 transforms=_processor):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.frames, self.features = load_data_csv(self.data_dir, pred=True)
        self.transforms = transforms
        self.transforms = self.processor_fit(transforms)
        self.frames = { station: pd.DataFrame(self.transforms.transform(_df),
                                        index=_df.index,
                                        columns=_df.columns) for station, _df in self.frames.items() }
        self.sequence_datasets = self.gen_sequence_datasets(self.frames, target=None)

    def processor_fit(self, preprocessor):
        for _, _df in self.frames.items():
             preprocessor.fit(_df)
        return preprocessor
    
    def process_preds(self, plist):
        plist = [l[-12:][0].squeeze(0) for l in plist]
        indexes = [generate_datetime_index(v.index.max(), periods=l.size(0)) for l, v in zip(plist, self.frames.values())]
        plist = [pd.DataFrame(self.transforms.inverse_transform(p.numpy()), index=i, columns=self.features) for i, p in zip(indexes, plist)]
        stations = list(self.frames.keys())
        preds = {}
        for s, p in zip(stations, plist):
            preds[s] = p
        return preds
    
    def gen_sequence_datasets(self, frames, target=None):
        sequence_datasets = {}
        for s, _df in frames.items():
            sequence_datasets[s] = SequenceDataset(_df,
                                                   s,
                                                   target=['precipitation'],
                                                   features=self.features)
        return sequence_datasets
             
    def gen_train_sets(self, dataset):
        split = int(len(dataset)*0.6)
        indices = np.arange(split)
        train_set = Subset(dataset, indices)
        return train_set
    
    def gen_val_loaders(self, dataset):
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

    def gen_test_loader(self, dataset):
        split = int(len(dataset)*0.8)
        indices = np.arange(split, len(dataset))
        test_loader = DataLoader(Subset(dataset, indices),
                                 batch_size=self.batch_size,
                                 drop_last=True,
                                 shuffle=False,
                                 collate_fn=collate_batch,
                                 num_workers=2)
        return test_loader

    def predict_combined_loader(self, preds=None):
        pred_loaders = {}
        if preds:
            datasets = self.gen_sequence_datasets(preds)
        else:
            datasets = self.sequence_datasets
        for s, _df in datasets.items():
            pred_loaders[s] = _df
        return CombinedLoader(pred_loaders, 'sequential')
        
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
