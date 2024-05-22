import httpx
import json
from os import path, getcwd
from src.prediction_models.rainfall_predictor.AllWeatherCubeLoader import loadJson


def loadJsonFixture():
    '''
    load the sample Json file to the Cube query resonse model format.
    Note: The query JSON contains:
    - station Ids [27]
    - date range: ["2020-03-05T00:00:00.000", "2020-03-12T23:59:59.999"]
    '''
    p = path.join(
        getcwd(), 'test',
        'fixtures',
        'all_weather_cube_query_response.json'
    )
    with open(p, 'r') as file:
        jsonData = json.load(file)
        return json.dumps(jsonData)


# https://lundberg.github.io/respx/
# import httpx
# import pytest


# def test_default(respx_mock):
#     respx_mock.get("https://foo.bar/").mock(return_value=httpx.Response(204))
#     response = httpx.get("https://foo.bar/")
#     assert response.status_code == 204


# @pytest.mark.respx(base_url="https://foo.bar")
# def test_with_marker(respx_mock):
#     respx_mock.get("/baz/").mock(return_value=httpx.Response(204))
#     response = httpx.get("https://foo.bar/baz/")
#     assert response.status_code == 204
