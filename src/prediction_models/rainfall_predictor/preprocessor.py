import json
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from tqdm.auto import tqdm
"""
Needed for imports from
from json and csv without
pydantic validation.    
"""

cols = [ 
    "station",
    "date",
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

def load_dataframe(df: pd.DataFrame, json=False) -> dict:
    if json:
        df = pd.json_normalize(df)
    df = df.reindex(columns=cols)
    df = duplicate_datetime(df.copy())
    df = set_dt_index(df.copy())
    df = df[(df.index.year.isin([2022, 2023]))].copy()
    df = negatives(df)
    df = outliers(df)
    df = impute_vals(df)
    return {s: _df.drop('station', axis=1) 
        for s, _df in df.groupby('station')}
    
def set_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("date", inplace=True)
    return df

def sample_interp(df, agg_dict):
    df = df.resample('5min').ffill(limit=12)
    num_cols = df.select_dtypes(include=np.number).columns.to_list()
    df[num_cols] = df[num_cols].interpolate(method='time', limit=12)
    return df.resample('h').agg(agg_dict)

def duplicate_datetime(df: pd.DataFrame, datetime_col="date") -> pd.DataFrame:
    df[datetime_col] = pd.to_datetime(df[datetime_col], format='mixed')
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
    mask = X[num_cols] > X[num_cols].quantile(0.9999)
    X[mask] = np.nan
    return X

def impute_vals(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    other = X.select_dtypes(exclude=np.number).columns.to_list()
    imputer = SimpleImputer(missing_values=np.nan)
    X[num_cols] = imputer.fit_transform(X[num_cols].values)
    return X
