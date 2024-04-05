import json
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

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

def load_data_json(strJsonPath: str, pred: bool = True):
    """
    Load json and set column names and order
    to what the pretrained model is expecting.
    """
    cols = [
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
    
    remap = {
        "all_weather.date": "date",
        "all_weather.station": "station",
        "all_weather.sum_precipitation": "precipitation",
        "all_weather.avg_temperature": "temperature",    
        "all_weather.avg_humidity": "humidity",
        "all_weather.avg_pressure": "pressure",
        "all_weather.avg_wind_speed": "wind_speed",
        "all_weather.avg_wind_direction": "wind_direction",
        "all_weather.avg_solar": "solar",
    }

    with open(strJsonPath, encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data=data['data'])
    if df.columns.to_list() != cols:
        df = df.rename(columns=remap)
        df = df[cols] 
    df = duplicate_datetime(df.copy())        
    df = set_dt_index(df.copy())
    features = df.columns.to_list()
    print(features)
    features = [ x for x in features if x != 'station']
    precip = transforms[1].fit_transform(df.precipitation.values.reshape(-1,1))
    df.precipitation = precip
    dfs = {s: _df.drop('station', axis=1) for s, _df in df.groupby('station')} 
    return {s: pd.DataFrame(transforms[0].fit_transform(_df.values), 
                            index=_df.index, 
                            columns=_df.columns) for s, _df in dfs.items()}, transforms, features 

def load_data_csv(data_dir: str,
                    pred: bool = False,
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
                    usecols=["station", "date", "precipitation", "temperature", "humidity", "pressure", "wind_speed", "wind_direction", "solar"],
                    dtype=type_dict,
                    skipinitialspace=True,
                    chunksize=1000,
                
                ),
                desc="Loading CSV data",
                total=lines_number // 1000 + 1,    
            )
        ]
    )
    return load_dataframe(df)

def load_dataframe(json_list: list[str]) -> dict:
    # df.drop(df.loc[df["station"] == '27'].index, inplace=True)
    return pd.json_normalize(json_list)
   # df = duplicate_datetime(df.copy())        
   # df = set_dt_index(df.copy())
   # return {s: _df.drop('station', axis=1) for s, _df in df.groupby('station')} 

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
    return X # [(np.abs(zscore(X[num_cols])) < 3).all(axis=1)]
    
# def generate_time_lags(df, target, n_lags):
#     df_n = df.copy()
#     for n in range(1, n_lags + 1):
#         lagged_targets = list(df_n[target].shift(n))
#         # We use concat here for performance reasons
#         df_n = pd.concat([df_n, pd.Series(lagged_targets, name=f"lag{n}",index=df_n.index)],axis=1)
#     # Remove the first n rows where no 'previous' target is attainable for the number of lags
#     df_n = df_n.iloc[n_lags:]
#     return df_n

# def cyclical_transformer(df, column, period, start_val, drop_original=True):
#     df[f"{column}_sin"] = np.sin(2 * np.pi * (df[column] - start_val) / period)
#     df[f"{column}_cos"] = np.cos(2 * np.pi * (df[column] - start_val) / period)
#     if drop_original:
#         df = df.drop(column, axis=1)
#     return df

# # The period and start values were determined from the min/max of the columns
# cyclical_features = {"day": (31,1), "month": (12,1), "day_of_week": (6,0), "week_of_year": (53,1)}
# for cyclical_feature, (period, start_val) in cyclical_features.items():
#     df_features = cyclical_transformer(df_features, cyclical_feature, period, start_val)
# df_features = (
#                 df_gen
#                 .assign(day = df_gen.index.day)
#                 .assign(month = df_gen.index.month)
#                 .assign(day_of_week = df_gen.index.dayofweek)
#                 .assign(week_of_year = df_gen.index.isocalendar().week)
#               )
