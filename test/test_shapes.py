
import json
from os import path, getcwd
from datetime import date
import numpy as np
import pandas as pd
from src.prediction_models.lstm_predictor.dataset import (gen_pred_dataset,
    standard_transform,
    onehot_transform,
    max_inverse_transform)
from src.prediction_models.lstm_predictor.preprocessor import load_dataframe
from src.prediction_models.lstm_predictor.utils import (reload_model, 
    plot_predictions,
    concatenate_latent_representation,
    compute_stochastic_dropout,
    rescale_predictions)
from src.prediction_models.lstm_predictor.AllWeatherConfig import getAllWeatherConfig
from src.interfaces.CubeJsQueryRequest import CubeQueryRequest
from src.prediction_models.rainfall_predictor.AllWeatherCubeRequest import makeAllWeatherQueryReq
from src.prediction_models.rainfall_predictor.PredictionPostRequests import CubePredictionPostRequest


if __name__ == "__main__":

    # get prediction window to test shapes
    # of model inputs and datasets
    config = getAllWeatherMLConfig()
    prediction_window = config.experiment_config.prediction_window
    # Load model checkpoints
    
    encoder = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')

    # test data
    with open('./tmp/all_weather_cube_query_response.json') as f:
        d = json.load(f)
        weather_data = d['data']
    # load test data
    data = load_dataframe(weather_data)
    pr_data = data[data.station == '61'].iloc[-700:-500, :]
    X_p, y_p = gen_pred_dataset(pr_data, prediction_window)
    # test shape of concatenated latent space `X_s_`
    X_x = standard_transform(X_p)
    X_o = onehot_transform(X_p)
    X_s = np.concatenate((X_x, X_o), axis=-1)
    y_s = max_inverse_transform(y_p)
    X_s_ = concatenate_latent_representation(encoder, X_s)

    
    
