
import json
import pandas as pd
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
"""
Needed for imports from
from json and csv without
pydantic validation.    
"""

cols = [ 
    "station",
  #  "date",
    "precipitation",
  #  "lag1",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "solar",
]

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
                    usecols=[x for x in cols if x != 'lag1'],
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

def load_dataframe(df: list | pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, list):
        df = pd.DataFrame(df)
    df.station = df.station.astype('str')
    df = duplicate_datetime(df)
    df = set_dt_index(df)
    df = negatives(df)
    df = outliers(df)
    df = impute_vals(df)
    df = df.reindex(columns=cols)
    df = rainy_season(df)
    return df
    
def set_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("date", inplace=True)
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
    mask = X[num_cols] > X[num_cols].quantile(0.999)
    X[mask] = np.nan
    return X

def impute_vals(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    other = X.select_dtypes(exclude=np.number).columns.to_list()
    imputer = SimpleImputer(missing_values=np.nan)
    X[num_cols] = imputer.fit_transform(X[num_cols].values)
    return X

def generate_time_lags(df: pd.DataFrame, 
                        value: str, 
                        n_lags: int) -> pd.DataFrame:
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        lagged_values = list(df_n[value].shift(n))
        
        df_n = pd.concat([pd.Series(lagged_values, name=f"lag{n}",index=df_n.index), df_n],axis=1)
    # Remove the first n rows where no 'previous' value is attainable for the number of lags
    df_n = df_n.iloc[n_lags:]
    return df_n

def rainy_season(df):
    mask = df.index.month.isin([12, 1, 2, 3, 4])
    df['rainy_season'] = mask.astype('int')
    return df
