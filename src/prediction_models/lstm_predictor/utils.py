import os
from os import path
import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

from sklearn.metrics import mean_absolute_error
from AllWeatherConfig import getAllWeatherMLConfig

config = getAllWeatherMLConfig()

prediction_window = config.experiment_config.prediction_window   # header?
pretrain_path = config.trainer_config.pretrained_path
accelerator = config.trainer_config.accelerator

if accelerator == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

device_name = tf.test.gpu_device_name()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def plot_predictions(preds: any, target: any = None):
    x = np.arange(len(preds))
    plt.plot(x, preds)
    if target:
        plt.plot(x, target)
    plt.savefig('./plots/predictions.pdf')

def reload_model(filename: str):
    p = path.abspath(path.join(pretrain_path, 
        filename))    
    model = tf.keras.models.load_model(p)
    if len(model.layers) > 2:
        for l in model.layers:
            l.trainable = False
    else:
        model.layers[0].trainable = False
    return model

def concatenate_latent_representation(encoder: any,
                                      X: np.array,
                                      y: np.array = None):
    with tf.device(device_name):
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

def jsonify_ndarray(arr: np.array): 
    return json.dumps(arr.tolist(), cls=NumpyEncoder)

def rescale_predictions(predictions: np.array):
    p = predictions.copy()
    _indices = p > p.max()*0.70
    p[_indices] = p[_indices]*3.2
    return p
