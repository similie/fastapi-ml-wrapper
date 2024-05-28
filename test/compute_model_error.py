from os import getcwd, path
import json
import numpy as np
import tqdm
from sklearn.metrics import mean_absolute_error
from src.prediction_models.rainfall_predictor.dataset import (
    gen_pred_dataset,
    standard_transform,
    onehot_transform,
    max_inverse_transform
)
from src.prediction_models.rainfall_predictor.preprocessor import load_dataframe
from src.prediction_models.rainfall_predictor.utils import (
    NumpyEncoder,
    reload_model,
    # plot_predictions,
    concatenate_latent_representation,
    # compute_stochastic_dropout,  # moved from utils, only used in this file
    # jsonify_ndarray,  # moved from utils, only used in this file
    # rescale_predictions
)
from src.prediction_models.rainfall_predictor.AllWeatherConfig import getAllWeatherMLConfig


def compute_stochastic_dropout(model: any, X_test, y_test):
    '''
    Cycle 20 through predictions with dropout to quantify error.
    See https://arxiv.org/pdf/1506.02142.pdf
    '''
    scores = []
    for i in tqdm(range(0, 20)):
        scores.append(mean_absolute_error(y_test, model.predict(X_test).ravel()))
    return scores, np.mean(scores), np.std(scores)


def jsonify_ndarray(arr: np.array):
    '''
    Convert numpy array to json.
    '''
    return json.dumps(arr.tolist(), cls=NumpyEncoder)


if __name__ == "__main__":
    config = getAllWeatherMLConfig()
    prediction_window = config.experiment_config.prediction_window

    p = path.join(getcwd(), config.inference_checkpoints)
    encoder = reload_model(path.join(p, 'encoder.keras'))
    fc_model = reload_model(path.join(p, 'forecaster.keras'))

    with open('./tmp/all_the_weather.json') as f:
        d = json.load(f)
        weather_data = d['data']

    data = load_dataframe(weather_data)
    X_p, y_p = gen_pred_dataset(data, prediction_window)

    X_x = standard_transform(X_p)
    X_o = onehot_transform(X_p)
    X_s = np.concatenate((X_x, X_o), axis=-1)
    y_s = max_inverse_transform(y_p)
    X_s_ = concatenate_latent_representation(encoder, X_s)

    scores, mean_error, std_error = compute_stochastic_dropout(fc_model, X_s_, y_s)

    payload = jsonify_ndarray(scores)

    print("\n\nMAE: ", mean_error.round(5), "\n")
    print("Standard Deviation:\n\n", std_error)
