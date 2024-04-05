from os import path, getcwd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import (Dataset, 
                            DataLoader, 
                            Subset, 
                            ConcatDataset)

from preprocessor import load_dataframe
from pytorch_lightning.utilities import CombinedLoader

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer, RobustScaler

processor = make_pipeline(RobustScaler())
power = make_pipeline(QuantileTransformer(n_quantiles=200))

transforms = [processor, power]

class SequenceDataset(Dataset):
    """
    Timeseries dataset 
    Args: 
    - target (leave as `None` for the AE)
 
    """
    def __init__(self, 
                 dataframe,
                 features,
                 target=None, 
                 prediction_window=12,
                 sequence_length=12):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.dataframe = dataframe
        if self.target == None:
            self.target = self.features
        self.X = self.dataframe[self.features] # .iloc[:-self.prediction_window]
        self.Y = self.dataframe[self.target].shift(
            periods=-self.prediction_window,
            freq='h')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if (idx+self.sequence_length) > len(self.dataframe):
            x, y = self.pad_df(idx)
        else:
            indexes = list(range(idx, idx + self.sequence_length))
            x = self.X.iloc[indexes, :].values
            y = self.Y.iloc[indexes, :].values
        return torch.tensor(x).float(), torch.tensor(y).float()

    def pad_df(self, idx):
        g = gap(idx)            
        indexes = list(range(idx, len(self.dataframe)))
        return to_pad(indexes, g)

    def gap(self, idx: int) -> int:
        return idx+self.sequence_length - len(self.dataframe)

    def to_pad(indexes: list, 
                gap: int) -> tuple[np.ndarray, np.ndarray]:
        x_to_pad = self.X.iloc[indexes, :]
        y_to_pad = self.Y.iloc[indexes, :]
        xpad = pd.concat([x_to_pad.iloc[:,0]]*gap).reset_index(drop=True)
        ypad = pd.concat([y_to_pad.iloc[:,0]]*gap).reset_index(drop=True)
        x = pd.concat([x_to_pad, xpad]).reset_index(drop=True).values
        y = pd.concat([y_to_pad, ypad]).reset_index(drop=True).values
        return x, y
                 
class data_module():
    """
    Data Module for time series forecasting

    """
    def __init__(self,
                 data: dict[str, pd.DataFrame],
                 features: list = ["precipitation", 
                                    "temperature", 
                                    "humidity", 
                                    "pressure", 
                                    "wind_speed", 
                                    "wind_direction",
                                    "solar"],
                 batch_size: int = 1,
                 transforms: list = transforms,
                 target: list | None = None):
        self.data_dir = data
        self.batch_size = batch_size
        self.features = features
        self.target = target
        self.frames = load_dataframe(self.data_dir)
        self.sequence_datasets = self.gen_sequence_datasets(self.frames, self.features)

    def setup(self, stage=None):
        if stage == None | "fit":
            self.train_dataloader =  self.train_combined_loader()
            self.val_dataloader = self.val_combined_loader()
            self.test_dataloader = self.test_combined_loader()
        if stage == "predict":
            self.predict_dataloader = self.predict_combined_loader()   
    # def process_preds(self, plist: list) -> dict[str, pd.DataFrame]:
    #     plist = [l[-12:][0].squeeze(0) for l in plist]
    #     indexes = [generate_datetime_index(v.index.max(), periods=l.size(0)) for l, v in zip(plist, self.frames.values())]
    #     plist = [pd.DataFrame(self.transforms[0].inverse_transform(p.numpy()), index=i, columns=self.features) for i, p in zip(indexes, plist)]
    #     plist = [pd.DataFrame(np.hstack((self.transforms[1].inverse_transform(p.values[:,0].reshape(-1,1)), p.values[:,1:])), index=i, columns=self.features) for i, p in zip(indexes, plist)]
    #     stations = list(self.frames.keys())
    #     preds = {}
    #     for s, p in zip(stations, plist):
    #         preds[s] = p
    #     return preds

    def process_preds(self, preds: list) -> dict[str, pd.DataFrame]:
        t_preds = self.truncate_predication_sequences(preds)
        t_trans = self.inverse_transform_pipeline(t_preds)
        station_idx = [(s, self.generate_datetime_index(_df.index.max()))
                            for s, _df in self.frames.items()]
        preds = {}
        for s, p in zip(station_idx, t_trans):
            preds[s[0]] = pd.DataFrame(p, columns=self.features, index=s[1])
        return preds 

    def truncate_prediction_sequences(self, preds: list) -> list:
        return [l[-12:][0].squeeze(0) for l in preds]

    def generate_datetime_index(self, start_time, periods=11):
        return pd.date_range(start=start_time, freq='h', periods=periods)
    
    def inverse_transform_pipeline(self, tvals: torch.tensor, idx: pd.date_range) -> pd.DataFrame:
        vals = self.transforms[0].inverse_transform(tvals.numpy())
        precip = self.transforms[1].inverse_transform(vals[:,0])
        return np.hstack((precip, vals[:,1:]))

    def transform_pipeline(self, df):
        if df.precipitation:
            precipitation = df['precipitation'].values.reshape(-1,1)
            df['precipitation'] = self.transforms[1].fit_transform(precipitation)
        vals = df.values
        return pd.DataFrame(self.transforms[0].fit_transform(vals),
                            columns=df.columns,
                            index=df.index) 

    def gen_sequence_datasets(self, frames: dict, features, target=None) -> dict[str, SequenceDataset]:
        sequence_datasets = {}
        for s, _df in frames.items():
            sequence_datasets[s] = SequenceDataset(_df,
                                                   s,
                                                   features=features,
                                                   target=target)
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
            datasets = self.gen_sequence_datasets(preds, self.features, self.target)
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

def collate_batch(batch):
    x = [item[-1] for item in batch]
    x = pad_sequence(x, batch_first=True)
    y = [item[0] for item in batch]
    y = pad_sequence(y, batch_first=True)
    return x, y
