import pytest
import respx
from httpx import Response
from json import loads
from fastapi.testclient import TestClient
from app.main import app
from src.prediction_models.rainfall_predictor.AllWeatherConfig import getAllWeatherConfig
from src.prediction_models.rainfall_predictor.AllWeatherCubeResponse import (
    AllWeatherCubeQueryResponse,
    cleanCubeNameFromResponseKeys
)
from test.fixtures.all_weather import loadJsonFixture
from src.prediction_models.rainfall_predictor.PredictionPostRequests import (
#     CubePredictionPostRequest,
    ForecastPredictionPostRequest
)

client = TestClient(app)


def test_get_model_template():
    res = client.get('/api/v1/template/rainfall_predictor')
    print(res)
    assert res.status_code == 200


def test_post_model_predict_wrong_payload():
    body = {"not_a_weather_forecast": 1}
    res = client.post('/api/v1/predict/rainfall_predictor', json=body)

    assert res.status_code == 400


@respx.mock
def test_post_model_predict_cube_payload():
    # CubePredictionPostRequest
    # We mock the GET request in AllWeatherCubeLoader->loadJson called by
    # RainfallPredictor.predict on: /api/v1/predict/<model_name>
    body = {
        "model": "rainfall_predictor",
        "dateRange": ['2024-06-01T00:00:01Z', '2024-06-02T23:59:59Z'],
        "stations": [27]
    }
    config = getAllWeatherConfig()
    baseUrl = config.cube_rest_api
    cubeResponse = loadJsonFixture()

    apiRoute = respx.get(baseUrl).mock(return_value=Response(200, text=cubeResponse))
    res = client.post('/api/v1/predict/rainfall_predictor', json=body)

    assert apiRoute.called is True
    assert res.status_code == 200
    assert isinstance(res.json()['data'], list)
    assert len(res.json()['data']) > 0


def test_post_model_predict_forecast_payload():
    # ForecastPredictionPostRequest
    # "precipitation", "temperature", "humidity", "pressure", "wind_speed", "wind_direction", "solar"
    # body = {
    #     "modelName": "rainfall_predictor",
    #     "data": [{
    #         "station": "27",
    #         "hour": "2020-03-05T01:00:00.000",
    #         "date": "2020-03-05T01:00:00.000",
    #         "wind_direction": 209.98000308275223,
    #         "wind_speed": 2.3660000205039977,
    #         "soil_moisture": 0,
    #         "dew_point": 0,
    #         "solar": 543.1,
    #         "temperature": 29.570000076293944,
    #         "humidity": 85.45770568847657,
    #         "pressure": 1008.0299926757813,
    #         "precipitation": 1.5
    #     }]
    # }
    cubeResponse = loadJsonFixture()
    jsonString = cleanCubeNameFromResponseKeys(cubeResponse)
    model = AllWeatherCubeQueryResponse.model_validate_json(jsonString)
    print(model.data[0].model_dump_json(by_alias=True))
    # data = []
    # for d in res.data:
    #     data.append(d.model_dump(by_alias=True))
    
    # req = {
    #     "modelName": "rainfall_predictor",
    #     "data": data
    # }

    res = client.post('/api/v1/predict/rainfall_predictor', json=loads(jsonString))

    print(res)
    assert res.status_code == 200
