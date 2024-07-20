import numpy as np
from .dataset import (gen_pred_dataset, standard_transform, onehot_transform)
from .preprocessor import load_dataframe
from .utils import (
    reload_model,
    concatenate_latent_representation,
    rescale_predictions
)
from .AllWeatherConfig import getAllWeatherMLConfig


config = getAllWeatherMLConfig()
prediction_window = config.experiment_config.prediction_window
accelerator = config.trainer_config.accelerator


def predict(weather_data: any, debug: bool = False) -> list[float]:
    """
    Load pretrained models and generate predictions
    from input data (json)
    """
    encoder = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')

    data = load_dataframe(weather_data)
    X_p, y_p = gen_pred_dataset(data, prediction_window)

    X_x = standard_transform(X_p)
    X_o = onehot_transform(X_p)
    X_s = np.concatenate((X_x, X_o), axis=-1)
    X_s_ = concatenate_latent_representation(encoder, X_s)

    predictions = fc_model.predict(X_s_)
    preds = rescale_predictions(predictions)

    if debug is True:   # pragma: no cover
        mse = ((preds - y_p)**2).mean(axis=0)
        print("\n\nMSE: ", mse[0].round(5), "\n")
        print("Summary:\n\n", data.describe())

    return preds.flatten().tolist()
