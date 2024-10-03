import json
import numpy as np
from pytest import approx
from src.prediction_models.rainfall_predictor.AllWeatherConfig import getAllWeatherMLConfig
from src.prediction_models.rainfall_predictor.preprocessor import load_dataframe
from src.prediction_models.rainfall_predictor.dataset import (
    gen_pred_dataset,
    standard_transform,
    onehot_transform
)
from src.prediction_models.rainfall_predictor.utils import (
    NumpyEncoder,
    reload_model,
    concatenate_latent_representation,
    rescale_predictions
)
from test.fixtures.all_weather import loadJsonFixture, serialiseToML


def test_load_models():
    model = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')

    assert model is not None
    assert fc_model is not None


def test_shapes():
    # Load model checkpoints
    encoder = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')

    # Note: 137 is valid for a 128 dimension trained model
    assert encoder.layers[0].batch_shape[1:] == (12, 9)
    assert fc_model.layers[0].batch_shape[1:] == (12, 137)


def test_numpy_encoder():
    npe = NumpyEncoder()

    arr = np.zeros(5, np.float16)
    s1 = json.dumps(arr.tolist(), cls=NumpyEncoder)
    s2 = json.dumps(arr, cls=NumpyEncoder)

    assert npe is not None
    assert len(s1) > 0
    assert len(s2) > 0


def test_concat_latent_representation():
    config = getAllWeatherMLConfig()
    prediction_window = config.experiment_config.prediction_window
    encoder = reload_model('encoder.keras')
    jsonData = loadJsonFixture()
    weather_data = serialiseToML(jsonData)
    data = load_dataframe(weather_data)
    X_p, y_p = gen_pred_dataset(data, prediction_window)

    X_x = standard_transform(X_p)
    X_o = onehot_transform(X_p)
    X_s = np.concatenate((X_x, X_o), axis=-1)
    X_s_ = concatenate_latent_representation(encoder, X_s)

    assert X_s_ is not None
    # Note: 137 is valid for a 128 dimension trained model
    assert X_s_.shape[1:] == (12, 137)


def test_rescale_predictions():
    arr = np.zeros(5, np.float16)
    arr[0] = 0.8
    arr[1] = 0.75
    arr[2] = 0.7
    arr[3] = 0.6
    # threshold for this array is 0.56
    arr[4] = 0.5
    res = rescale_predictions(arr)

    assert res[0] == approx(2.56, rel=1e-3)
    assert res[1] == approx(2.40, rel=1e-3)
    assert res[2] == approx(2.24, rel=1e-3)
    assert res[3] == approx(1.92, rel=1e-3)
    assert res[4] == approx(0.5)
