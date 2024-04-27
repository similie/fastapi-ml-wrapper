
import json
from os import path, getcwd
import pytest
from ..src.interfaces.ReqRes import BasePostRequest
from ..src.prediction_models.AllModelsService import (
    getModelNames,
    modelForPayload,
    ensureValidModelName
)
from ..src.prediction_models import RainfallPredictor
from ..src.prediction_models.rainfall_predictor.\
    PredictionPostRequests import (ForecastPredictionPostRequest,
    CubePredictionPostRequest)

from ..src.prediction_models.\
    rainfall_predictor.AllWeatherConfig import (getAllWeatherMLConfig,
    getAllWeatherConfig)

from ..src.prediction_models.rainfall_predictor.\
    PredictionPostRequests import ForecastPredictionPostRequest
from ..src.prediction_models.rainfall_predictor.\
    AllWeatherCubeResponse import (cleanCubeNameFromResponseKeys,
    AllWeatherCubeQueryResponse)


def loadJsonFixture():
    '''
    load the sample Json file to the Cube query resonse model format.
    '''
    p = path.join(getcwd(),
        'test',
        'fixtures', 
        'all_weather_cube_query_response.json')
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

def test_predict():
    """
    Provide a json object of sample test data to
    test the predict pathway.
    """
    d = serialise_to_ml()
    # req = CubePredictionPostRequest.model_validate_json(d)
    predictor = RainfallPredictor
    # payload = [item.model_dump(by_alias=True) for item in req.data]
    predictions = predictor.predict(weather_data=d)
    assert predictor is not None
    assert predictions is not None
