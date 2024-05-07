import json
from os import path, getcwd
from pathlib import Path
from src.prediction_models.rainfall_predictor.AllWeatherCubeResponse import (
    cleanCubeNameFromResponseKeys,
    AllWeatherCubeQueryResponse
)
from src.prediction_models.rainfall_predictor.dataset import (
    pathForPretrainedPickle,
    standard_transform,
    max_transform,
    onehot_transform,
    gen_pred_dataset,
    # TODO: max_inverse_transform,
    # TODO: standard_inverse_transform
)
from src.prediction_models.rainfall_predictor.preprocessor import load_dataframe


def loadJsonFixture():
    '''
    load the sample Json file to the Cube query resonse model format.
    '''
    p = path.join(
        getcwd(),
        'test',
        'fixtures',
        'all_weather_cube_query_response.json'
    )
    with open(p, 'r') as file:
        jsonData = json.load(file)
        return json.dumps(jsonData)


def serialise_to_ml():
    jsonData = loadJsonFixture()
    # cubeName = getAllWeatherConfig().cube_name
    cleanedJson = cleanCubeNameFromResponseKeys(jsonData)
    jsonData = json.loads(cleanedJson)
    model = AllWeatherCubeQueryResponse.model_validate(jsonData)
    return model.model_dump(by_alias=True)['data']


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
    data = serialise_to_ml()
    weather_data = load_dataframe(data)
    X, y = gen_pred_dataset(weather_data, 12)
    X_s = standard_transform(X)
    X_o = onehot_transform(X)
    y_s = max_transform(y)
    # X_inv = standard_inverse_transform(X_s)
    # y_inv = max_inverse_transform(y_s)

    assert X_s is not None
    assert X_o is not None
    assert y_s is not None

    # assert X_inv == X
    # assert y_inv == y
