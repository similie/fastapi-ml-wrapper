import numpy as np
import pandas as pd
from dataset import (gen_pred_dataset,
                    standard_transform,
                    onehot_transform,
                    max_inverse_transform)
from utils import (reload_model, 
                    plot_predictions,
                    concatenate_latent_representation)

if __name__ == "__main__":


    encoder = reload_model('./pretrained_checkpoints/encoder.keras')
    fc_model = reload_model('./pretrained_checkpoints/forecaster.keras')

    with open('./all_the_weather.json') as f:
        d = json.load(f)
        weather_data = d['data']

    data = load_dataframe(weather_data)

    X_p, y_p = gen_pred_dataset(pr_data, 12)

    X_x = standard_transform(X_p)
    X_o = onehot_transform(X_p)
    X_s = np.concatenate((X_x, X_o), axis=-1)

    X_s_ = concatenate_latent_representation(encoder, X_s)

    predictions = fc_model.predict(X_s_)
    preds = max_inverse_transform(predictions)

    payload = jsonify_ndarray(preds)

    mse = ((preds - y_s)**2).mean(axis=0)

    print("\n\nMSE: ", mse[0].round(5), "\n")
    print("Summary:\n", df.describe())
