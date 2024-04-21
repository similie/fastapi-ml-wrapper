import json
import numpy as np
np.float = float # hack for multiflow
import pandas as pd 
from utils import load_knn, pickle_knn, jsonify_ndarray
from preprocessor import load_dataframe
from data import (max_transform, 
                    standard_transform, 
                    max_inverse_transform,
                    standard_inverse_transform,
                    onehot_transform,
                    groupdf)
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluatePrequential
import knn
from knn import SAMKNNRegressor
import matplotlib.pyplot as plt


if __name__ == "__main__":
# jsonEndPoint with weather forecast data here

    samKNN  = load_knn('./pickle/sam_knn.pkl')
    samKNN.print_model()

    with open('./tmp/all_the_weather.json') as f:
        d = json.load(f)
        weather_data = d['data']

    df = load_dataframe(weather_data)
    pr = df[df.station == '61'].iloc[-700:-500, :]

    X = df.drop(['precipitation', 'station'], axis=1).values
    y = df['precipitation'].values
    assert len(X) == len(y)
    print(X.shape, y.shape)

    assert len(X_s) == len(y_s)

    Z_s = standard_transform(X)
    R_s = onehot_transform(X)
    X_s = np.concatenate((Z_s, R_s), axis=-1)
    y_z = max_transform(y)

    # Init Data Stream from np.array
    ds = DataStream(X_s, y=y_s)
    ds.prepare_for_use()

    predictions = samKNN.predict(X_s)
    preds = max_inverse_transform(predictions)

    mse = ((preds - y_s)**2).mean(axis=0)

    print("\n\nMSE: ", mse[0].round(5), "\n")
    print("Summary:\n", df.describe())

    payload = jsonify_ndarray(preds)


