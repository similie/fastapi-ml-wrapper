
import json
from os import path, getcwd
from datetime import date
import numpy as np
import pandas as pd
from src.prediction_models.lstm_predictor.dataset import (gen_pred_dataset,
    standard_transform,
    onehot_transform,
    max_transform,
    max_inverse_transform)
from src.prediction_models.rainfall_predictor.preprocessor import load_dataframe
from src.prediction_models.rainfall_predictor.utils import (reload_model, 
    plot_predictions,
    concatenate_latent_representation,
    compute_stochastic_dropout,
    rescale_predictions)
from src.prediction_models.lstm_predictor.AllWeatherConfig import getAllWeatherMLConfig

if __name__ == "__main__":

   # get prediction window to test shapes
    # of model inputs and datasets
    config = getAllWeatherMLConfig()
    prediction_window = config.experiment_config.prediction_window
    
    # test data
    with open('./tmp/all_weather_cube_query_response.json') as f:
        d = json.load(f)
        weather_data = d['data']
    # load test data
    data = load_dataframe(weather_data)
    data.describe()
    data.columns
