import json
from os import path, getcwd
from src.prediction_models.rainfall_predictor.AllWeatherConfig import getAllWeatherConfig
from src.prediction_models.rainfall_predictor.AllWeatherCubeResponse import AllWeatherCubeQueryResponse
from src.prediction_models.rainfall_predictor.AllWeatherCubeResponse import cleanCubeNameFromResponseKeys


def loadJsonFixture():
    '''
    load the sample Json file to the Cube query resonse model format.
    '''
    p = path.join(getcwd(), 'test', 'fixtures', 'all_weather_cube_query_response.json')
    with open(p, 'r') as file:
        jsonData = json.load(file)
        return json.dumps(jsonData)


def test_clean_cubename_from_response():
    jsonData = loadJsonFixture()
    assert jsonData is not None

    cubeName = getAllWeatherConfig().cube_name
    cleanedJson = cleanCubeNameFromResponseKeys(jsonData)
    assert cleanedJson.find(cubeName) == -1

    jsonData = json.loads(cleanedJson)
    model = AllWeatherCubeQueryResponse.model_validate(jsonData)
    assert model is not None
    assert len(model.data) > 0
