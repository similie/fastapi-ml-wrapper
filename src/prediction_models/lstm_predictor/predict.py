import os
import numpy as np
import pandas as pd
from .dataset import (gen_pred_dataset,
                    standard_transform,
                    onehot_transform,
                    max_inverse_transform)
from .preprocessor import load_dataframe
from .utils import (reload_model, 
                    plot_predictions,
                    concatenate_latent_representation,
                    rescale_predictions)
from .AllWeatherConfig import getAllWeatherMLConfig

config = getAllWeatherMLConfig()
prediction_window = config.experiment_config.prediction_window
accelerator = config.trainer_config.accelerator

if accelerator == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def predict(weather_data):
    encoder = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')
    
    data = load_dataframe(weather_data)
    X_p, y_p = gen_pred_dataset(weather_data, prediction_window)
    
    X_x = standard_transform(X_p)
    X_o = onehot_transform(X_p)
    X_s = np.concatenate((X_x, X_o), axis=-1)
    X_s_ = concatenate_latent_representation(encoder, X_s)

    predictions = fc_model.predict(X_s_)
    preds = rescale_predictions(predictions)
    mse = ((preds - y_p)**2).mean(axis=0)

    print("\n\nMSE: ", mse[0].round(5), "\n")
    print("Summary:\n\n", data.describe())

    payload = jsonify_ndarray(preds)
    return payload
