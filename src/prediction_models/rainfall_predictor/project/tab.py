import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from math import ceil
from os import path
from .process import field_list, duplicate_datetime, outliers, set_dt_index

class TabulaTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Get_data():
    def __init__(self, data_path, 
                 groupby_col, 
                 target_col, 
                 prediction_window,
                 sequence_length,
                 batch_size, 
                 df=None,
                 flag='json'):
        self.data_path = data_path
        self.groupby_col = groupby_col
        self.target_col = target_col
        self.prediction_window = prediction_window
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.yscaler = MinMaxScaler()
        self.xscaler = StandardScaler()
        self.df = df
        self.flag = flag
        # if not self.df:
        #     # if self.flag == "csv":
        #     #     self.df = load_data_csv(self.data_path, type_dict=type_dict, field_list=field_list, prediction_window=self.prediction_window)
        #     # if self.flag == "json":
        #     print(f'loading json, self.dataPAth is: {self.data_path}')
        #     jsonPath = path.join(getcwd(), "data/all_weather_cube_query_response.json")
        #     # print(jsonPath)

        #     self.df = self.load_data_json(jsonPath)
        #     print(f'In Get_data()->Init: df.cols: {self.df.columns}')
        #     print(f'In Get_data()->Init: df.value_counts: {self.df["station"].value_counts()}')
            

    def __prepStationsAndDataSets(self):
        self.stations = self.df[self.groupby_col].value_counts().index.to_list()
        self.df = [self.shift_target(_df.drop(self.groupby_col, axis=1), self.target_col, self.prediction_window) for _, _df in self.df.groupby(self.groupby_col)]
        self.sets = []
        for _df in self.df:
            X, y = _df.drop(self.target_col, axis=1).values, _df[self.target_col].values.reshape(-1,1)
            X, y = self.xscaler.fit_transform(X), self.yscaler.fit_transform(y)
            X, y = self.seq_list(X, self.sequence_length), self.seq_list(y, self.sequence_length)
            self.sets.append(TabulaTensorDataset(X, y))
        
    def make_stage(self, stage):    
        if stage == "fit":
            train_loaders = []
            val_loaders = []
            for dd in self.sets:
                length = len(dd)
                split = int(length*0.8)
                tr_name = Subset(dd, range(split))
                val_name = Subset(dd, range(split, length))
                train_loaders.append(DataLoader(tr_name, batch_size=self.batch_size, shuffle=False, num_workers=4))
                val_loaders.append(DataLoader(val_name, batch_size=self.batch_size, shuffle=False, num_workers=4))
            self.train, self.val = train_loaders, val_loaders
        if stage == "test":
            test_stations = []
            for dd in self.sets:
                length = len(dd)
                split = int(length*0.9)
                test_name = Subset(dd, range(split, length))
                test_stations.append(DataLoader(test_name, self.batch_size, shuffle=False, num_workers=4))
            self.test = test_stations
        if stage == "predict":
            pred_stations = []
            for dd in self.sets:
                pred_stations.append(DataLoader(dd, self.batch_size, shuffle=False, num_workers=0))
            self.pred = pred_stations
        
    def shift_target(self, df, target_col, split_idx):
        df[target_col] = df[target_col].shift(periods=split_idx, fill_value=0)
        return df
    
    def seq_list(self, X: np.ndarray, seq_length: int):
        X = [X[idx*seq_length:idx*seq_length+seq_length] for idx in range(ceil(len(X)/seq_length))]
        return X

    def load_data_json(self, jsonFilePath: str):
        """
        load sensor data from the specified file path. Handles duplicate
        datetimes, sets datetime index.
        """
        df = pd.read_json(jsonFilePath)
        return self.__refineLoadedJson(df)


    def setJsonData(self, data: pd.DataFrame):
        '''
        Adds the pre-loaded Json data from the dataframe into the model
        '''
        return self.__refineLoadedJson(data)


    def __refineLoadedJson(self, dataframe: pd.DataFrame):
        '''
        Post process a raw Json load to remove any columns that we're not
        processing and run the basic data organisation for scale/fit
        '''
        cols = dataframe.columns.to_list()
        # remove data columns that the ML Model is not expecting
        for col in cols:
            if not field_list.__contains__(col):
                dataframe.drop(col, axis=1, inplace=True)

        dataframe = duplicate_datetime(dataframe)
        dataframe = set_dt_index(dataframe)
        dataframe = outliers(dataframe, 7)  # TODO: add the offset days to config

        self.df = dataframe
        self.__prepStationsAndDataSets()
        return self
