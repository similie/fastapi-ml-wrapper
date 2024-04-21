from os import path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, 
                                    MaxAbsScaler,
                                    OneHotEncoder)
def groupdf(df):
    grp = df.groupby('station')
    res = []
    for s, _df in grp:
        res.append(_df)
    return pd.concat(res)

def max_transform(y: np.array):
    mx_path = './pickle/maxabs.pkl'
    if path.isfile(mx_path):
        maxabs = pickle.load(open(mx_path, 'rb'))
        y_s = maxabs.transform(y.reshape(-1,1))
        return y_s
    else:
        maxabs = MaxAbsScaler()
        y_s = maxabs.fit_transform(y.reshape(-1,1))
        pickle.dump(maxabs, open(mx_path, 'wb'))
        return y_s     

def onehot_transform(R: np.array):
    oh_path = './pickle/onehot.pkl'
    if path.isfile(oh_path):
        onehot = pickle.load(open(oh_path, 'rb'))
        R_s = onehot.transform(R[:,-1:]).toarray()
        return R_s
    else:
        onehot = OneHotEncoder()
        R_s = onehot.fit_transform(R[:,-1:]).toarray()
        pickle.dump(onehot, open(oh_path, 'wb'))
        return R_s
    
def standard_transform(X: np.array):
    st_path = './pickle/standard.pkl'
    if path.isfile(st_path):
        standard = pickle.load(open(st_path, 'rb'))
        X_s = standard.transform(X[:,:-1])
        return X_s
    else:
        standard = StandardScaler()
        X_s = standard.fit_transform(X[:,:-1])
        pickle.dump(standard, open(st_path, 'wb'))
        return X_s  
       
def max_inverse_transform(y: np.array):
    mx_path = './pickle/maxabs.pkl'
    if path.isfile(mx_path):
        maxabs = pickle.load(open(mx_path, 'rb'))
        y_s = maxabs.inverse_transform(y.reshape(-1,1))
        return y_s
    else:
        print("Scaler not fitted.")
        return y

def standard_inverse_transform(X: np.array):
    st_path = './pickle/standard.pkl'
    if path.isfile(st_path):
        standard = pickle.load(open(st_path, 'rb'))
        X_s = standard.inverse_transform(X[:,:-2])
        return X_s
    else:
        print("Scaler not fitted.")
        return X
