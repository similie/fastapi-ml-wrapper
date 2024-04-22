import numpy as np
import pandas as pd
from dataset import (gen_pred_dataset,
                    standard_transform,
                    onehot_transform,
                    max_inverse_transform)
from preprocessor import load_dataframe
from utils import (reload_model, 
                    plot_predictions,
                    concatenate_latent_representation)
from AllWeatherConfig import getAllWeatherConfig


if __name__ == "__main__":

    config = getAllWeatherConfig() # should I instantiate this in the
                                   # header?
    encoder = reload_model('./pretrained_checkpoints/encoder.keras')
    fc_model = reload_model('./pretrained_checkpoints/forecaster.keras')

    with open('./tmp/all_weather_cube_query_response.json') as f:
        d = json.load(f)
        weather_data = d['data']

    data = load_dataframe(weather_data)
    X_p, y_p = gen_pred_dataset(data, 12)

    X_x = standard_transform(X_p)
    X_o = onehot_transform(X_p)
    X_s = np.concatenate((X_x, X_o), axis=-1)
    y_s = max_inverse_transform(y_p)
    X_s_ = concatenate_latent_representation(encoder, X_s)

    predictions = fc_model.predict(X_s_)
    preds = max_inverse_transform(predictions)

    payload = jsonify_ndarray(preds)

    mse = ((preds - y_s)**2).mean(axis=0)

    print("\n\nMSE: ", mse[0].round(5), "\n")
    print("Summary:\n\n", df.describe())
