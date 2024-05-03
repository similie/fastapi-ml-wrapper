import json
import numpy as np
from src.prediction_models.rainfall_predictor.dataset import (
    gen_pred_dataset,
    standard_transform,
    onehot_transform,
    max_inverse_transform
)
from src.prediction_models.rainfall_predictor.preprocessor import load_dataframe
from src.prediction_models.rainfall_predictor.utils import (
    reload_model,
    # plot_predictions,
    concatenate_latent_representation,
    compute_stochastic_dropout,
    jsonify_ndarray,
    # rescale_predictions
)
from src.prediction_models.rainfall_predictor.AllWeatherConfig import getAllWeatherConfig


if __name__ == "__main__":
    config = getAllWeatherConfig()  # should I instantiate this in the
    prediction_window = config.experiment_predicion_window   # header?

    encoder = reload_model('./pretrained_checkpoints/encoder.keras')
    fc_model = reload_model('./pretrained_checkpoints/forecaster.keras')

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
