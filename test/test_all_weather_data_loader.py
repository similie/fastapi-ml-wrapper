import httpx
from httpx import Response
import respx
from src.prediction_models.rainfall_predictor.AllWeatherConfig import getAllWeatherConfig
from src.prediction_models.rainfall_predictor.AllWeatherCubeLoader import loadJson
from test.fixtures.all_weather import loadJsonFixture


@respx.mock
def test_decorator():
    '''Demo function with: https://lundberg.github.io/respx/'''
    my_route = respx.get("http://localhost:5002/foo")
    response = httpx.get("http://localhost:5002/foo")
    assert my_route.called
    assert response.status_code == 200


@respx.mock
def test_load_all_weather_cube():
    config = getAllWeatherConfig()
    baseUrl = config.cube_rest_api
    cubeResponse = loadJsonFixture()

    apiRoute = respx.get(baseUrl).mock(return_value=Response(200, text=cubeResponse))
    res = loadJson(["2020-03-05T00:00:00.000", "2020-03-12T23:59:59.999"], [27])
    # Note: `res` is an AllweatherCubeQueryResponse (not an httpx.Response)
    assert apiRoute.called
    assert res.total is not None
    assert res.data is not None
    assert res.total == len(res.data)
