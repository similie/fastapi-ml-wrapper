from os import path
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler,
                                   RobustScaler,
                                   MinMaxScaler,
                                   MaxAbsScaler)
from .AllWeatherConfig import getAllWeatherMLConfig

config = getAllWeatherMLConfig()

features = config.experiment_config.features
target = config.experiment_config.target_col

def gen_sequence(df, sequence_length, seq_cols):
    numerical_data = df[seq_cols].values
    num_elements = numerical_data.shape[0]

    for start, stop in zip(range(0, num_elements-sequence_length), range(sequence_length, num_elements)):
        yield numerical_data[start:stop, :]

def gen_labels(df, sequence_length, label):
    numerical_data = df[label].values
    num_elements = numerical_data.shape[0]
    return numerical_data[sequence_length:num_elements, :]

def gen_pred_dataset(df, sequence_length: int):

    X_pred, y_pred = [], []

    for station, _df in df.groupby(["station"]):
        for seq in gen_sequence(_df, sequence_length, features):
            X_pred.append(seq)

        for seq in gen_labels(_df, sequence_length, target):
            y_pred.append(seq)

    X_pred = np.asarray(X_pred)
    y_pred = np.asarray(y_pred)
    return X_pred, y_pred

def groupdf(df):
    grp = df.groupby('station')
    res = []
    for s, _df in grp:
        res.append(_df)
    return pd.concat(res)

def max_transform(y: np.array):
    mx_path = path.join('./pickle', 'maxabs.pkl')
    if path.isfile(mx_path):
        maxabs = pickle.load(open(mx_path, 'rb'))
        y_s = maxabs.transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
        return y_s
    else:
        maxabs = MaxAbsScaler()
        maxabs.fit(y.reshape(-1, y.shape[-1]))
        pickle.dump(maxabs, open(mx_path, 'wb'))
        return y_s     

def onehot_transform(R_: np.array):
    R = R_[:,:,-1:].astype('int')
    ohe = np.eye(2)[R.reshape(-1)]
    return ohe.reshape((R.shape[0], R.shape[1], R.shape[-1]+1))

def standard_transform(X_: np.array):
    st_path = path.join('./pickle', 'standard.pkl')
    X = X_[:, :, :-1] # remove rainy_season
    if path.isfile(st_path):
        standard = pickle.load(open(st_path, 'rb'))
        X_s = standard.transform(X.reshape(-1,X.shape[-1])).reshape(X.shape)
        return X_s
    else:
        standard = StandardScaler()
        standard.fit(X.reshape(-1,X.shape[-1]))
        pickle.dump(standard, open(st_path, 'wb'))
        return X_s  
       
def max_inverse_transform(y: np.array):
    mx_path = path.join('./pickle', 'maxabs.pkl')
    if path.isfile(mx_path):
        maxabs = pickle.load(open(mx_path, 'rb'))
        y_s = maxabs.inverse_transform(y.reshape(-1,1))
        return y_s
    else:
        print("Scaler not fitted.")
        return y

def standard_inverse_transform(X: np.array):
    st_path = path.join('./pickle', 'standard.pkl')
    if path.isfile(st_path):
        standard = pickle.load(open(st_path, 'rb'))
        X_s = standard.inverse_transform(X[:,:-2])
        return X_s
    else:
        print("Scaler not fitted.")
        return X
    
