from pathlib import Path
from src.prediction_models.rainfall_predictor.dataset import (
    # TODO: gen_sequence,
    # TODO: gen_labels,
    gen_pred_dataset,
    # TODO: groupdf,
    pathForPretrainedPickle,
    max_transform,
    onehot_transform,
    standard_transform,
    max_inverse_transform,
    # deprecated. not used in this project: standard_inverse_transform
)
from src.prediction_models.rainfall_predictor.preprocessor import load_dataframe
from test.fixtures.all_weather import loadJsonFixture, serialiseToML


def test_pickle_file_loader():
    pklFile = pathForPretrainedPickle('standard.pkl')
    p = Path(pklFile)

    assert p.exists() is True
    assert p.is_file() is True


def test_non_existant_pickle_file_loader():
    pklFile = pathForPretrainedPickle('standard')
    p = Path(pklFile)

    assert p.exists() is False


def test_scalers():
    """
    Provide a json object of sample test data to
    test the scaler functions.
    """
    jsonData = loadJsonFixture()
    data = serialiseToML(jsonData)
    weather_data = load_dataframe(data)
    X, y = gen_pred_dataset(weather_data, 12)
    X_s = standard_transform(X)
    X_o = onehot_transform(X)
    y_s = max_transform(y)
    # X_inv = standard_inverse_transform(X_s)
    y_inv = max_inverse_transform(y_s)

    assert X_s is not None
    assert X_o is not None
    assert y_s is not None

    # assert X_inv == X
    assert y_inv is not None
