import json
import pandas as pd
import numpy as np

from AllWeatherConfig import AllWeatherMLConfig

from sklearn.impute import SimpleImputer

from tqdm.auto import tqdm
"""
Needed for imports from
from json and csv without
pydantic validation.    
"""

config = AllWeatherMLConfig()
groupby_col = config.experiment_config.groupby_col
cols = groupby_col + ["date"] + config.experiment_config.features

agg_dict = {
    "station": "ffill",
    "precipitation": "sum",
    "temperature": "mean",
    "humidity": "mean",
    "pressure": "mean",
    "wind_speed": "mean",
    "wind_direction": "mean",
    "solar": "mean"
}

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

def load_data_csv(data_dir: str,
                    cols: list[str] = cols,
                    type_dict: dict = type_dict):
    with open(data_dir) as f:
        lines_number = sum(1 for _ in f)
        f.close()
    df = pd.concat(
        [
            chunk
            for chunk in tqdm(
                pd.read_csv(
                    data_dir,
                    usecols=cols,
                    dtype=type_dict,
                    skipinitialspace=True,
                    chunksize=1000,
                ),
                desc="Loading CSV data",
                total=lines_number // 1000 + 1,    
            )
        ]
    )
    df = df[df.station != '27']
    return df

def load_dataframe(df: list | pd.DataFrame) -> dict:
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    df = df.reindex(columns=cols)
    df[groupby_col] = df[groupby_col].astype('str')
    df = duplicate_datetime(df.copy())
    df = set_dt_index(df.copy())
    df = df[(df.index.year.isin([2020, 2021, 2022, 2023, 2024]))].copy()
    df = negatives(df)
    df = outliers(df)
    df = sample_interp(df, agg_dict)
    df = impute_vals(df)
    df = df.dropna()
    # return df
    return {s[0]: _df.drop(groupby_col, axis=1) 
        for s, _df in df.groupby(groupby_col) if len(_df) >= 11}
    
def set_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("date", inplace=True)
    return df

def sample_interp(df, agg_dict):
    df = df.resample('15min').first()
    df = df.resample('h').agg(agg_dict)
    df.index = df.index.to_period('h')
    return df

def duplicate_datetime(df: pd.DataFrame, datetime_col="date") -> pd.DataFrame:
    df[datetime_col] = pd.to_datetime(df[datetime_col], 
                                        format='mixed',
                                        utc=True)
    delta = pd.to_timedelta(df.groupby("date").cumcount(), unit="ms")
    df.date = df.date + delta.values
    df.sort_values("date", inplace=True)
    return df

def negatives(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    mask = X[num_cols] < 0
    X[mask] = np.nan
    return X

def outliers(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    mask = X[num_cols] > X[num_cols].quantile(0.99)
    X[mask] = np.nan
    return X

def impute_vals(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    other = X.select_dtypes(exclude=np.number).columns.to_list()
    imputer = SimpleImputer(missing_values=np.nan)
    X[num_cols] = imputer.fit_transform(X[num_cols].values)
    return X
