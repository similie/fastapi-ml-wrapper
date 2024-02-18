#!/usr/bin/env python

"""process.py: preprocess the data selecting relevant fields
identifying outliers (isolation forest) and imputing missing values."""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

# TODO replace

## plotting
# import matplotlib
# import seaborn as sns
# from pylab import rcParams
# import matplotlib.pyplot as plt
# from matplotlib import rc

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

# sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

# sns.set_palette(sns.color_palette(COLORS_PALETTE))

# rcParams["figure.figsize"] = 12, 8

# Utilities for data preprocessing
knn_imputer = KNNImputer(n_neighbors=3)
forest = IsolationForest()
current_directory = os.getcwd()
csv_path = os.path.join(current_directory, "data/ordered_csv/")
csv_filename = "combined.csv"

# for resampling df
agg_dict = {
    "station": "ffill",
    "precipitation": np.sum,
    "temperature": np.mean,
    "humidity": np.mean,
    "pressure": np.mean,
    "wind_speed": np.mean,
    "wind_direction": np.mean,
    "solar": np.mean,
    "month": "ffill",
    "season": "ffill",
    "anomaly": "ffill",
}

# Initial import fields
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

# Utilities

# change forest anomaly to bool
zero_to_one = np.vectorize(lambda x: 1 if x == -1 else 0)


def duplicate_datetime(df, datetime_col="date"):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    delta = pd.to_timedelta(df.groupby("date").cumcount(), unit="ms")
    df.date = df.date + delta.values
    df.sort_values("date", inplace=True)
    return df


def set_dt_index(df):
    df.set_index("date", inplace=True)
    return df


def negatives(df):
    num_cols = df.select_dtypes(include=np.number).columns.to_list()
    df["sensor_anomaly"] = 0
    mask = df[num_cols] < 0
    df["sensor_anomaly"] = mask
    df[mask] = np.nan
    return df


def outliers(X, num_std):
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    mask = X[num_cols] < 0
    X[mask] = np.nan
    X["sensor_anomaly"] = 0
    for col in num_cols:
        if col != "temperature":
            mask = X[
                ((X[col] - X[col].mean(axis=0)) / X[col].std(axis=0) > num_std)
            ].index
            X.loc[mask, "sensor_anomaly"] = 1
            X.loc[mask, col] = np.nan
        else:
            mask = X[
                ((X[col] - X[col].mean(axis=0)) / X[col].std(axis=0) > (num_std - 4))
            ].index
            X.loc[mask, "sensor_anomaly"] = 1
            X.loc[mask, col] = np.nan
    return X


class DataframeFunctionTransformer:
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df, **transform_params)

    def fit(self, X, y=None, **fit_params):
        return self


def load_data_csv(path, type_dict=type_dict, field_list=field_list):
    """
    load sensor data from csv file in chunks,
    handles duplicate datetimes, sets
    datetime index, supply a dictionary of
    types and a list of db field names.
    """
    df = pd.concat(
        [
            chunk
            for chunk in tqdm(
                pd.read_csv(
                    path,
                    usecols=field_list,
                    dtype=type_dict,
                    skipinitialspace=True,
                    chunksize=1000,
                ),
                desc="Loading CSV data",
                total=4000,
            )
        ]
    )
    df.drop(df.loc[df["station"] == 27].index, inplace=True)
    df = DataframeFunctionTransformer(duplicate_datetime).transform(
        df, datetime_col="date"
    )
    df = DataframeFunctionTransformer(set_dt_index).transform(df)
    df = df[(df.index.year.isin([2022, 2023]))].copy()
    df = DataframeFunctionTransformer(outliers).transform(df)
    return df.dropna()


df = load_data_csv(
    os.path.join(csv_path, csv_filename), type_dict=type_dict, field_list=field_list
)
