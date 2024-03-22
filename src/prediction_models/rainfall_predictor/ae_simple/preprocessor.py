import pandas as pd
import numpy as np
import json
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from dateutil.relativedelta import relativedelta
# Add train and test flags for time deltas eg.

def monthdelta(date, delta):
    return date + relativedelta(months=delta)

def daydelta(date, delta):
    return date + relativedelta(days=delta)

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
zero_to_one = np.vectorize(lambda x: 0 if x == -1 else 1)

iso_transform = make_column_transformer(
    (IsolationForest(random_state = 0, contamination = float(0.005)), ['precipitation']),
    (IsolationForest(random_state = 0, contamination = float(0.001)), ['humidity']),
    (IsolationForest(random_state = 0, contamination = float(0.001)), ['wind_speed']),
    (IsolationForest(random_state = 0, contamination = float(0.001)), ['solar']),
    (IsolationForest(random_state = 0, contamination = float(0.001)), ['temperature']),
    (IsolationForest(random_state = 0, contamination = float(0.001)), ['pressure']),
    (IsolationForest(random_state = 0, contamination = float(0.001)), ['wind_direction']),
    remainder='passthrough'
)

impute = SimpleImputer(strategy='median', missing_values=np.NaN)
pipeline = make_pipeline(iso_transform)

qt_processor = make_column_transformer(
    (QuantileTransformer(n_quantiles=5, random_state=0), ['precipitation']),
    (StandardScaler(), ['humidity', 'wind_speed', 'solar', 'temperature', 'pressure', 'wind_direction']),
    remainder='passthrough'
)

preprocessor = make_pipeline(qt_processor)

def qt_transform(df, processor=preprocessor):
    return pd.DataFrame(preprocessor.fit_transform(df),
                        columns=df.columns, index=df.index)

def inverse_qt_transform(df, processor=preprocessor):
    return preprocessor.inverse_transform(df)

def imputer(df, imputer=impute):
    return pd.DataFrame(imputer.fit_transform(df), 
                        columns=df.columns, index=df.index)
    
def iso_pipeline(df, pipeline=pipeline):
    pipeline['columntransformer'].transformers[0][1].fit(df)
    pipeline['columntransformer'].transformers[1][1].fit(df)
    pipeline['columntransformer'].transformers[2][1].fit(df)
    pipeline['columntransformer'].transformers[3][1].fit(df)
    pipeline['columntransformer'].transformers[4][1].fit(df)
    pipeline['columntransformer'].transformers[5][1].fit(df)
    pipeline['columntransformer'].transformers[6][1].fit(df)

    d1 = zero_to_one(pipeline['columntransformer'].transformers[0][1].decision_function(df))
    d2 = zero_to_one(pipeline['columntransformer'].transformers[1][1].decision_function(df))
    d3 = zero_to_one(pipeline['columntransformer'].transformers[2][1].decision_function(df))
    d4 = zero_to_one(pipeline['columntransformer'].transformers[3][1].decision_function(df))
    d5 = zero_to_one(pipeline['columntransformer'].transformers[4][1].decision_function(df))
    d6 = zero_to_one(pipeline['columntransformer'].transformers[5][1].decision_function(df))
    d7 = zero_to_one(pipeline['columntransformer'].transformers[6][1].decision_function(df))
    a1 = zero_to_one(pipeline['columntransformer'].transformers[0][1].predict(df))
    a2 = zero_to_one(pipeline['columntransformer'].transformers[1][1].predict(df))
    a3 = zero_to_one(pipeline['columntransformer'].transformers[2][1].predict(df))
    a4 = zero_to_one(pipeline['columntransformer'].transformers[3][1].predict(df))
    a5 = zero_to_one(pipeline['columntransformer'].transformers[4][1].predict(df))
    a6 = zero_to_one(pipeline['columntransformer'].transformers[5][1].predict(df))
    a7 = zero_to_one(pipeline['columntransformer'].transformers[6][1].predict(df))
    a1 = a2 | a3 | a4 | a5 | a6 | a7
    df.loc[df.index, 'anomaly'] = a1.copy()
    idf = df[df.anomaly == 1].copy()
    idf.drop('anomaly', axis=1, inplace=True)
    return idf
    
def load_data_csv(data_dir,
                    pred=True,
                    type_dict=type_dict):

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
    df.drop(df.loc[df["station"] == '27'].index, inplace=True)
    df = duplicate_datetime(df.copy())        
    df = set_dt_index(df.copy())
    df = df[(df.index.year.isin([2022, 2023]))].copy()
    df = negatives(df.copy())
    df = iso_pipeline(df.dropna(), pipeline)
    df = sample_interp(df.copy(), agg_dict=agg_dict)
    df = df.dropna()
    if pred:        
        df = df[(df.index >= (daydelta(df.index.max(), -268))) & (df.index <= (daydelta(df.index.max(), -100)))] #lkg
    return {s: _df.drop('station', axis=1) for s, _df in df.groupby('station')}, df.drop('station', axis=1).columns.to_list()

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

def load_dataframe(df: pd.DataFrame):
    
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
    df = df[cols]
    df = duplicate_datetime(df.copy())        
    df = set_dt_index(df.copy())
    # df = df[(df.index.year.isin([2022, 2023]))].copy()
    df = negatives(df.copy())
    # df = iso_pipeline(df.dropna(), pipeline)
    return {s: _df.drop('station', axis=1) for s, _df in df.groupby('station')}, df.drop('station', axis=1).columns.to_list()

def load_data_json(strJsonPath: str):
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
    
    data = []

    with open(strJsonPath, encoding='utf-8') as f:
        data = json.load(f)
        # lines_number = sum(1 for _ in f)
        # for line in tqdm(f, desc="Loading Json...", total=lines_number*10):
        #     doc = json.loads(line)
        #     lst = [doc[col] for col in cols]
        #     data.append(lst)

    df = pd.DataFrame(data=data, columns=cols)
        # "date",
    # df = duplicate_datetime(df.copy())        
    # df = set_dt_index(df.copy())
    # # df = df[(df.index.year.isin([2022, 2023]))].copy()
    # df = negatives(df.copy())
    # # df = iso_pipeline(df.dropna(), pipeline)
    # df = sample_interp(df.copy(), agg_dict=agg_dict)
    #df = df.dropna()
    # return {s: _df.drop('station', axis=1) for s, _df in df.groupby('station')}, df.drop('station', axis=1).columns.to_list()
    return df
