#!/usr/bin/env python

"""process.py: preprocess the data selecting relevant fields
identifying outliers (isolation forest) and imputing missing values."""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from dateutil.relativedelta import relativedelta
# Add train and test flags for time deltas eg.

def monthdelta(date, delta):
    return date + relativedelta(months=delta)

def daydelta(date, delta):
    return date + relativedelta(days=delta)

csv_root = Path('.').resolve().parent
csv_path = Path(csv_root, 'data/combined.csv').resolve()

# for resampling df
agg_dict = {
    "station": "ffill",
    "precipitation": "sum",
    "temperature": "mean",
    "humidity": "mean",
    "pressure": "mean",
    "wind_speed": "mean",
    "wind_direction": "mean",
    "solar": "mean",
}

# Initial import fields. TODO: from imported data, they're the model keys.
field_list = [
    "date",
    "station",
    "precipitation",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "solar",
]

type_dict = {
    "station": str,
    "date": str,
    "precipitation": float,
    "temperature": float,
    "humidity": float,
    "pressure": float,
    "wind_speed": float,
    "wind_direction": float,
    "solar": float,
}


def sample_interp(df, agg_dict):
    '''
    TODO: doc strings
    TODO: tests
    '''
    df = df.resample('5min').ffill(limit=30)
    num_cols = df.select_dtypes(include=np.number).columns.to_list()
    df[num_cols] = df[num_cols].interpolate(method='time', limit=12)
    return df.resample('h').agg(agg_dict)


def duplicate_datetime(df: pd.DataFrame, datetime_col="date") -> pd.DataFrame:
    '''
    TODO: doc strings
    TODO: tests
    '''
    df[datetime_col] = pd.to_datetime(df[datetime_col], format='mixed')
    delta = pd.to_timedelta(df.groupby("date").cumcount(), unit="ms")
    df.date = df.date + delta.values
    df.sort_values("date", inplace=True)
    return df


def set_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    '''
    TODO: doc strings
    TODO: tests
    '''
    df.set_index("date", inplace=True)
    return df


def negatives(df: pd.DataFrame) -> pd.DataFrame:
    '''
    TODO: doc strings
    TODO: tests
    '''
    num_cols = df.select_dtypes(include=np.number).columns.to_list()
    df["sensor_anomaly"] = 0
    mask = df[num_cols] < 0
    df["sensor_anomaly"] = mask
    df[mask] = np.nan
    return df


def outliers(X: pd.DataFrame, num_std: int) -> pd.DataFrame:
    '''
    TODO: doc strings
    TODO: tests
    '''
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    mask = X[num_cols] < 0
    X[mask] = np.nan

    return X


def load_data_csv(path, type_dict=type_dict, field_list=field_list, prediction_window=None):
    """
    load sensor data from csv file in chunks,
    handles duplicate datetimes, sets
    datetime index, supply a dictionary of
    types and a list of db field names.
    TODO: tests
    """
    with open(path) as f:
        lines_number = sum(1 for _ in f)
        
    df = pd.concat(
        [
            chunk
            # TODO: tdqm pointless in a webservice, maybe raise callback events later
            for chunk in tqdm(
                pd.read_csv(
                    path,
                    usecols=field_list,
                    dtype=type_dict,
                    skipinitialspace=True,
                    chunksize=1000,
                    
                ),
                desc="Loading CSV data",
                total=lines_number // 1000 + 1,    
            )
        ]
    )
    df.drop(df.loc[df["station"] == '27'].index, inplace=True)
    df = duplicate_datetime(df)
    df = set_dt_index(df)
    df = outliers(df, num_std=12)
    df = sample_interp(df, agg_dict)
    if prediction_window:
        df = df[df.index >= (daydelta(df.index.max(), -prediction_window))]
    
    return df.dropna()
