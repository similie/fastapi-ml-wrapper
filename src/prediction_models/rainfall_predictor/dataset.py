from os import path, getcwd
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler,
                                   MinMaxScaler,
                                   MaxAbsScaler)
from .AllWeatherConfig import getAllWeatherMLConfig

config = getAllWeatherMLConfig()

features = config.experiment_config.features + ['rainy_season']
target = config.experiment_config.target_col
pretrain_path = config.trainer_config.pretrained_path


def gen_sequence(df, sequence_length, seq_cols):
    numerical_data = df[seq_cols].values
    num_elements = numerical_data.shape[0]

    for start, stop in zip(range(0,
        num_elements-sequence_length),
        range(sequence_length, num_elements)):
        yield numerical_data[start:stop, :]


def gen_labels(df, sequence_length, label):
    numerical_data = df[label].values
    num_elements = numerical_data.shape[0]
    return numerical_data[sequence_length:num_elements, :]


def gen_pred_dataset(df, sequence_length: int):
    """
    Generate the dataset for predictions with historical
    and forcast data in the correct sequence length, ie:
    (batches, sequence_length, n_features)
    """
    X_pred, y_pred = [], []

    for station, _df in df.groupby(["station"]):
        for seq in gen_sequence(df, sequence_length, features):
            X_pred.append(seq)

        for seq in gen_labels(df, sequence_length, target):
            y_pred.append(seq)

    X_pred = np.asarray(X_pred)
    y_pred = np.asarray(y_pred)
    return X_pred, y_pred


def groupdf(df):
    """
    Remove, only needed for training.
    """
    grp = df.groupby('station')
    res = []
    for s, _df in grp:
        res.append(_df)
    return pd.concat(res)


def max_transform(y: np.array):
    """
    Loads the fitted normalizer to scale the
    input data target to unit scale.
    """
    p = path.abspath(path.join(getcwd(),
        pretrain_path,
        '..',
        'pickle',
        'maxabs.pkl')) 
    if path.isfile(p):
        maxabs = pickle.load(open(p, 'rb'))
        y_s = maxabs.transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
        return y_s
    else:
        maxabs = MaxAbsScaler()
        maxabs.fit(y.reshape(-1, y.shape[-1]))
        pickle.dump(maxabs, open(p, 'wb'))
        return y_s


def onehot_transform(R_: np.array):
    """
    Replaces the rainy_season column with a
    two-column one hot encoded feature. Rainy
    season must be the last feature.
    """
    R = R_[:, :, -1:].astype('int')
    ohe = np.eye(2)[R.reshape(-1)]
    return ohe.reshape((R.shape[0], R.shape[1], R.shape[-1]+1))


def standard_transform(X_: np.array):
    """
    Loads fitted standardscaler for the input
    data.
    """
    p = path.abspath(path.join(getcwd(),
        pretrain_path,
        '..',
        'pickle',
        'standard.pkl'))    
    X = X_[:, :, :-1]      
    if path.isfile(p):
        standard = pickle.load(open(p, 'rb'))
        X_s = standard.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return X_s
    else:
        standard = StandardScaler()
        standard.fit(X.reshape(-1, X.shape[-1]))
        pickle.dump(standard, open(p, 'wb'))
        return X_s  


def max_inverse_transform(y: np.array):
    """
    Inverse scale the target, currently not
    used in favor of our rescale_predictions
    function.
    """
    p = path.abspath(path.join(getcwd(),
        pretrain_path,
        '..',
        'pickle',
        'maxabs.pkl'))
    if path.isfile(p):
        maxabs = pickle.load(open(p, 'rb'))
        y_s = maxabs.inverse_transform(y.reshape(-1, 1))
        return y_s
    else:
        print("Scaler not fitted.")
        return y


def standard_inverse_transform(X: np.array):
    """
    Inverse transform features, removed one
    hot encoded rainy_season data. 
    """
    p = path.abspath(path.join(getcwd(),
        pretrain_path,
        '..',
        'pickle',
        'standard.pkl'))    
    if path.isfile(p):
        standard = pickle.load(open(p, 'rb'))
        X_s = standard.inverse_transform(X[:, :-2])
        return X_s
    else:
        print("Scaler not fitted.")
        return X
