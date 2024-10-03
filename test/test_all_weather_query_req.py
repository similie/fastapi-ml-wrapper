import json
from os import path, getcwd
from datetime import date
from src.interfaces.CubeJsQueryRequest import CubeQueryRequest
from src.prediction_models.rainfall_predictor.AllWeatherCubeRequest import makeAllWeatherQueryReq
from src.prediction_models.rainfall_predictor.PredictionPostRequests import CubePredictionPostRequest


def loadJsonCubeRequestFixture(getFilteredVersion: bool):
    # IMPORTANT. These terms need to align with the query json file names:
    # 'all_weather_cube_query_req_notfiltered'
    # 'all_weather_cube_query_req_filtered'
    s = 'filtered' if getFilteredVersion is True else 'notfiltered'
    p = path.join(
        getcwd(),
        'test',
        'fixtures',
        f'all_weather_cube_query_req_{s}.json'
    )
    with open(p, 'r') as file:
        jsonData = json.load(file)
        return jsonData  # json.dumps(jsonData)


def cubePostReqFixture():
    '''
    Returns station id, start date and end date to make a new instance
    of a CubePredictionPostRequest
    '''
    return 27, "2020-03-05", "2020-03-12"


def test_make_filtered_all_weather_query():
    json = loadJsonCubeRequestFixture(True)
    stationId, startDate, endDate = cubePostReqFixture()

    postReq = CubePredictionPostRequest(
        dateRange=[startDate, endDate],
        stations=[stationId]
    )
    assert postReq is not None

    cubeReq = makeAllWeatherQueryReq(postReq.dateRange, postReq.stations)
    assert cubeReq is not None

    cubeReqFixture = CubeQueryRequest.model_validate(json)
    assert cubeReqFixture is not None

    assert len(cubeReq.query.dimensions) == len(cubeReqFixture.query.dimensions)
    assert len(cubeReq.query.dimensions) > 0
    assert cubeReq.query.filters[0].values[0] == stationId
    assert cubeReq.query.timeDimensions[0].dateRange[0] == date.fromisoformat(startDate)
    assert cubeReq.query.timeDimensions[0].dateRange[1] == date.fromisoformat(endDate)


def test_make_non_filtered_all_weather_query():
    json = loadJsonCubeRequestFixture(False)
    stationId, startDate, endDate = cubePostReqFixture()

    postReq = CubePredictionPostRequest(
        dateRange=[startDate, endDate],
        stations=[]
    )
    assert postReq is not None

    cubeReq = makeAllWeatherQueryReq(postReq.dateRange, postReq.stations)
    assert cubeReq is not None

    cubeReqFixture = CubeQueryRequest.model_validate(json)
    assert cubeReqFixture is not None

    assert len(cubeReq.query.dimensions) == len(cubeReqFixture.query.dimensions)
    assert len(cubeReq.query.dimensions) == 0
    assert len(cubeReq.query.filters) == 0
    assert cubeReq.query.timeDimensions[0].dateRange[0] == date.fromisoformat(startDate)
    assert cubeReq.query.timeDimensions[0].dateRange[1] == date.fromisoformat(endDate)
