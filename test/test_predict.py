import json
from os import path, getcwd
from src.prediction_models.rainfall_predictor.predict import predict
from src.prediction_models.rainfall_predictor.AllWeatherCubeResponse import (
    cleanCubeNameFromResponseKeys,
    AllWeatherCubeQueryResponse
)


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
    cleanedJson = cleanCubeNameFromResponseKeys(jsonData)
    jsonData = json.loads(cleanedJson)
    model = AllWeatherCubeQueryResponse.model_validate(jsonData)
    return model.model_dump(by_alias=True)['data']


def test_predict():
    """
    Provide a json object of sample test data to
    test the predict pathway.
    """
    d = serialise_to_ml()
    predictions = predict(weather_data=d)
    assert predictions is not None
