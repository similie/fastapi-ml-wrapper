import os
from os import path, getcwd
import json
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from .AllWeatherConfig import getAllWeatherMLConfig


config = getAllWeatherMLConfig()
prediction_window = config.experiment_config.prediction_window
pretrain_path = config.trainer_config.pretrained_path
accelerator = config.trainer_config.accelerator

if accelerator == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

device_name = tf.test.gpu_device_name()


class NumpyEncoder(json.JSONEncoder):
    '''
    Utility class for jsonify function to convert numpy array to json
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# def plot_predictions(preds: any, target: any = None):
#     """
#     Plot predictions against target (optional)
#     """
#     x = np.arange(len(preds))
#     plt.plot(x, preds)
#     if target:
#         plt.plot(x, target)
#     plt.savefig('./plots/predictions.pdf')


def reload_model(filename: str):
    '''
    load .keras model checkpoint from the pretrain_path set in the config.
    If prefix folder is specified (e.g. for tests) it will be prepended to the
    pretrain_path, before constructing the final path.
    '''
    p = path.join(getcwd(), config.inference_checkpoints, filename)
    model = tf.keras.models.load_model(p)
    if len(model.layers) > 2:
        for layer in model.layers:
            layer.trainable = False
    else:
        model.layers[0].trainable = False
    return model


def concatenate_latent_representation(encoder: any, X: np.array, y: np.array = None):
    '''
    Concatenate the encoder layer 1 output with the
    features for the forecaster. Should only need the
    `X` array for the current model.
    '''
    with tf.device(device_name):
        X_ = np.concatenate([X, encoder.predict(X)], axis=-1)
        if isinstance(y, np.ndarray):  # pragma: no cover
            y_ = np.concatenate([y, encoder.predict(y)], axis=-1)
            return X_, y_
        return X_


def rescale_predictions(predictions: np.array):
    '''
    Apply a renormilization factor to the model
    predictions.
    '''
    p = predictions.copy()
    _indices = p > p.max()*0.70
    p[_indices] = p[_indices]*3.2
    return p
