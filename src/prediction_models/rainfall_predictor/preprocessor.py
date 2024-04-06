
import json
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

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
    json_lst = []
    for i in range(len(df)): 
        d = json.loads(df.iloc[i, :].to_json())
        json_lst.append(d)
    return json_lst

def load_dataframe(json_list: list[str]) -> dict:
    df = pd.json_normalize(json_list)
    df = df[cols].copy()
    df = duplicate_datetime(df.copy())        
    df = set_dt_index(df.copy())
    return {s: _df.drop('station', axis=1) for s, _df in df.groupby('station')} 
    
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
    X = X.dropna()
    return X
