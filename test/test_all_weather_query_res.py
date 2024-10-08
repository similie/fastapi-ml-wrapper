import json
from src.prediction_models.rainfall_predictor.AllWeatherConfig import getAllWeatherConfig
from src.prediction_models.rainfall_predictor.AllWeatherCubeResponse import AllWeatherCubeQueryResponse
from src.prediction_models.rainfall_predictor.AllWeatherCubeResponse import cleanCubeNameFromResponseKeys
from test.fixtures.all_weather import loadJsonFixture


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


def test_serialise_measures_to_ml():
    jsonData = loadJsonFixture()
    cleanedJson = cleanCubeNameFromResponseKeys(jsonData)
    jsonData = json.loads(cleanedJson)
    model = AllWeatherCubeQueryResponse.model_validate(jsonData)
    assert model is not None

    item = model.data[0]
    assert item is not None

    # without serialisation alias
    d = item.model_dump()
    assert d.get('avg_wind_direction') is not None
    assert d.get('wind_direction') is None

    # with serialisation alias
    d = item.model_dump(by_alias=True)
    assert d.get('avg_wind_direction') is None
    assert d.get('wind_direction') is not None
