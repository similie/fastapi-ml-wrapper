import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def plot_predictions(preds: any, target: any = None):
    x = np.arange(len(preds))
    plt.plot(x, preds)
    plt.plot(x, target)
    plt.savefig('./plots/predictions.pdf')

def reload_model(path: str):
  return tf.keras.models.load_model(path)

def concatenate_latent_representation(encoder: any,
                                      X: np.array,
                                      y: np.array = None):
    # uncomment and indent to use GPU
    with tf.device('/device:GPU:0'):
      X_ = np.concatenate([X,
        encoder.predict(X)], axis=-1)
      if isinstance(y, np.ndarray):
        y_ = np.concatenate([y,
          encoder.predict(y)], axis=-1)
        return X_, y_
      return X_

def compute_stochastic_dropout(model: any, X_test, y_test):
  scores = []
  for i in tqdm(range(0,20)):
    scores.append(mean_absolute_error(y_test, model.predict(X_test_).ravel()))
  return scores, np.mean(scores), np.std(scores)

def jsonify_ndarray(arr: np.array, 
    path: str = './predictions/preds.json'):
    return json.dumps(arr.tolist(), cls=NumpyEncoder)
