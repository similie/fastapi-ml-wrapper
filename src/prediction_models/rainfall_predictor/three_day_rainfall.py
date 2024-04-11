import sys
from os import path
import json

from predict import _predict
from dataset import data_module
from mutils import (get_checkpoint_filepath, 
                    plot_predictions,
                    serialise_ml_data)


if __name__ == "__main__":

    check_path = get_checkpoint_filepath(latent_dim=64)
    data = serialise_ml_data()
    weather_data = data.model_dump(by_alias=True)['data']
    predictions = _predict(weather_data, check_path)
    plot_predictions(predictions)
    
# pdf saved to the results folder root
