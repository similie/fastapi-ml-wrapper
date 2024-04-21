import json
import numpy as np
import pickle
from os import path
from sklearn.preprocessing import (StandardScaler, 
                                    MaxAbsScaler)
import knn
from knn import SAMKNNRegressor

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_knn(pkl_path: str):
    if path.isfile(pkl_path):
        model = unpickle_knn(pkl_path)
        return model
    else:
        print('No model found, loading model...')
        model = SAMKNNRegressor()
        return model
    
def pickle_knn(model: any, path: str = './pickle/sam_knn.pkl'):
  knn_file = open(path, 'wb') # open in binary
  pickle.dump(model, knn_file)
  knn_file.close()

def unpickle_knn(path: str = './pickle/sam_knn.pkl'):
   model = pickle.load(open(path, 'rb'))
   return model

# web service code ? with joblib
# model = joblib.load('scoreregression.pkl' , mmap_mode ='r')
def jsonify_ndarray(arr: np.array, 
    path: str = './predictions/preds.json'):
    return json.dumps(arr.tolist(), cls=NumpyEncoder)
