from os import path, getcwd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (Dataset, 
                            DataLoader, 
                            Subset, 
                            ConcatDataset)
from preprocessor import load_dataframe
from pytorch_lightning.utilities import CombinedLoader

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (PowerTransformer, 
    StandardScaler, 
    QuantileTransformer, 
    RobustScaler,
    MinMaxScaler)

class SequenceDataset(Dataset):
    """
    Timeseries dataset 
    Args: 
    - target (leave as `None` for the AE)
    - set target as precipitation for forecasting 
    """
    def __init__(self, 
                 dataframe,
                 features: list = ["precipitation", 
                                    "temperature", 
                                    "humidity", 
                                    "pressure", 
                                    "wind_speed", 
                                    "wind_direction",
                                    "solar"],
                 target=None, 
                 prediction_window=12,
                 sequence_length=12):
        self.features = features
        self.target = target
        if self.target == None:
            self.target = self.features
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.dataframe = dataframe
        self.X, self.Y = self.frame_to_sequence(self.dataframe)
        
    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.Y[idx]).float()
    
    def frame_to_sequence(self, dataframe: pd.DataFrame) -> np.array:
        x = []
        y = []
        X = dataframe[self.features][:-self.prediction_window]
        Y = dataframe[self.target].shift(-self.prediction_window)[:-self.prediction_window]
        if len(self.target) == 1:
            for idx in range(len(X)):
                xx = X.iloc[idx:idx+self.sequence_length,:].values
                yy = Y.iloc[idx:idx+self.sequence_length,:].values.reshape(-1,1)
                x.append(xx)
                y.append(yy)
        else:
            for idx in range(len(X)):
                xx = X.iloc[idx:idx+self.sequence_length,:].values
                yy = Y.iloc[idx:idx+self.sequence_length,:].values
                x.append(xx)
                y.append(yy)
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
                 target: list | None = None):
        self.batch_size = batch_size
        self.data = load_dataframe(data)
        self.features = features
        self.target = target
        self.transforms = [RobustScaler(), 
            QuantileTransformer(n_quantiles=200)]
        self.frames = self.frame_scale(self.data)
        self.sequence_datasets = self.gen_sequence_datasets(self.frames, 
                                                            self.target)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataloader = self.train_combined_loader()
            self.val_dataloader = self.val_combined_loader()
            self.test_dataloader = self.test_combined_loader()
        if stage == "predict":
            self.predict_dataloader = self.predict_combined_loader()  

    def process_preds(self, preds: list) -> dict[str, pd.DataFrame]:
        t_preds = self.truncate_prediction_sequences(preds)
        station_idx = [(s, self.generate_datetime_index(_df.index.max()))
                            for s, _df in self.frames.items()]
        t_trans = self.inverse_transform_pipeline(t_preds)
        preds = {}
        for s, p in zip(station_idx, t_trans):
            preds[s[0]] = pd.DataFrame(p, columns=self.features, index=s[1])
        return preds 

    def truncate_prediction_sequences(self, preds: list) -> list:
        return [l[-12:][0].squeeze(0) for l in preds]

    def generate_datetime_index(self, start_time, periods=12):
        return pd.date_range(start=start_time, freq='h', periods=periods)
    
    def inverse_transform_pipeline(self, 
                                   tvals: list[torch.tensor]) -> np.array:
        results = []
        for t in tvals:
            v =self.transforms[0].inverse_transform(t.numpy())
            p = self.transforms[1].inverse_transform(v[:,0].reshape(-1,1))
            results.append(np.hstack((p, v[:,1:])))
        return results
    
    def frame_scale(self, frames: dict):
        result = {}
        for s, _df in frames.items():
            result[s] = self.transform_pipeline(_df)
        return result
    
    def transform_pipeline(self, df: pd.DataFrame):
        if "precipitation" in df.columns.to_list():
            precip = df['precipitation'].values.reshape(-1,1)
            df['precipitation'] = self.transforms[1].fit_transform(precip)
        vals = df.values
        return pd.DataFrame(self.transforms[0].fit_transform(vals),
                            columns=df.columns,
                            index=df.index) 

    def gen_sequence_datasets(self, frames: dict, target) -> dict[str, SequenceDataset]:
        sequence_datasets = {}
        for s, _df in frames.items():
            sequence_datasets[s] = SequenceDataset(_df,
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

    def predict_combined_loader(self, preds=None):
        """
        Generates a prediction step dataloader
        from the sequence datasets of the data_module 
        class, a dictionary with station ids and 
        dataframes with weather data.
        """
        pred_loaders = {}
        if preds:
            datasets = self.gen_sequence_datasets(preds, target=['precipitation'])
        else:
            datasets = self.sequence_datasets
        for s, _df in datasets.items():
            pred_loaders[s] = _df
        return CombinedLoader(pred_loaders, 'sequential') # try max_size_cycle
        # pred_loaders = {}
        # datasets = self.sequence_datasets
        # for s, _df in datasets.items():
        #     pred_loaders[s] = _df
        # return CombinedLoader(pred_loaders, 'sequential') # try max_size_cycle
        
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
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, yy_pad


def test_datamodule(df):

    dm = data_module(data=df)
    dm.setup(stage='fit')
    train_loader = dm.train_dataloader
    val_loader = dm.val_dataloader
    test_loader = dm.test_dataloader

    tr_it = iter(train_loader)
    ts_it = iter(test_loader)
    vl_it = iter(val_loader)
    for it in [tr_it, ts_it, vl_it]:
        for i, (batch) in enumerate(it):
            if len(batch) == 2:
                inputs, target = batch
                if (torch.isnan(inputs).any() | torch.isnan(inputs).any()):
                    print("Is nan!")
            else:
                inputs, target = batch[0][0], batch[0][1]
                if (torch.isnan(inputs).any() | torch.isnan(inputs).any()):
                    print("Is nan!")