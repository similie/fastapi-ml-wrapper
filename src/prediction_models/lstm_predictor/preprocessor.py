import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from .AllWeatherConfig import getAllWeatherMLConfig

config = getAllWeatherMLConfig()
features = config.experiment_config.features
target = config.experiment_config.target_col


def load_dataframe(df: list | pd.DataFrame) -> pd.DataFrame:
    """
    Ingest json or dataframe for predictions.
    """
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if "station" in df.columns.to_list():
        df['station'] = df['station'].astype('str')
    df = duplicate_datetime(df.copy())
    df = set_dt_index(df.copy())
    df = negatives(df)
    df = outliers(df)
    df = impute_vals(df)
    df = df.reindex(columns=features)
    df = rainy_season(df)
    df = df.dropna()
    return df


def set_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date column to datetime and set it as 
    index.
    """
    df['date'] = pd.to_datetime(df['date'])
    df.set_index("date", inplace=True)
    return df


def duplicate_datetime(df: pd.DataFrame, datetime_col="date") -> pd.DataFrame:
    """
    Space out duplicate date times by an added millisecond. 
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col],
        format='mixed')

    delta = pd.to_timedelta(df.groupby("date").cumcount(), unit="ms")
    df.date = df.date + delta.values
    df.sort_values("date", inplace=True)
    return df


def negatives(X: pd.DataFrame) -> pd.DataFrame:
    """
    Removes negative anomalous sensor values.
    """
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    mask = X[num_cols] < 0
    X[mask] = np.nan
    return X


def outliers(X: pd.DataFrame) -> pd.DataFrame:
    """
    Removes extreme outlier values (+0.999 quantile)
    """
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    mask = X[num_cols] > X[num_cols].quantile(0.999)
    X[mask] = np.nan
    return X


def impute_vals(X: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with simple interpolation. 
    Shouldn't be needed.
    """
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    imputer = SimpleImputer(missing_values=np.nan)
    X[num_cols] = imputer.fit_transform(X[num_cols].values)
    return X


def rainy_season(df):
    """
    Indicate rainy season or not.
    """
    mask = df.index.month.isin([12, 1, 2, 3, 4])
    df['rainy_season'] = mask.astype('int')
    return df
