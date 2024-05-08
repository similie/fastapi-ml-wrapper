from datetime import datetime
from ...interfaces.CubeJsQueryResponse import (
    QueryMeasuresResponse,
    CubeQueryResponse
)
from .AllWeatherCubeRequest import AllWeatherQueryMeasures
from .AllWeatherConfig import getAllWeatherConfig


class AllWeatherQueryMeasuresResponse(AllWeatherQueryMeasures, QueryMeasuresResponse):
    '''
    All-weather fields additionally returned from a successful response. Station
    will be present if it was included as a dimension in the source query. E.g.
    setting a dimension of station and multiple station Ids in a filter will
    result in a dataset with each aggregated value per date per station.
    '''
    # station: int | None = None # inherited
    hour: datetime  # ISO date
    # date: datetime  # ISO date (inherited)


class AllWeatherCubeQueryResponse(CubeQueryResponse):
    '''
    Subclass overrides the type of the data in the data[list].
    '''
    data: list[AllWeatherQueryMeasuresResponse]

    # Example response with a station dimension included in the request
    # [{...},
    # {
    #   "all_weather.station": 27,
    #   "all_weather.date.hour": "2020-03-05T00:00:00.000",
    #   "all_weather.date": "2020-03-05T00:00:00.000",
    #   "all_weather.avg_wind_direction": 212.10892705300026,
    #   "all_weather.avg_wind_speed": 2.093035712838173,
    #   "all_weather.avg_soil_moisture": null,
    #   "all_weather.avg_dew_point": null,
    #   "all_weather.avg_solar": 512.4821428571429,
    #   "all_weather.avg_temperature": 29.301785673413956,
    #   "all_weather.avg_humidity": 83.37076595851353,
    #   "all_weather.avg_pressure": 1007.9321430751255,
    #   "all_weather.sum_precipitation": 0
    # },
    # {...}]


def cleanCubeNameFromResponseKeys(res: str) -> str:
    '''
    Removes the cube name and other illegal JSON key names from the response,
    e.g. date.hour. Note: depending on the aggregation time period used in the cube
    definition this may need to change to one of the values of TimeGranularity.
    '''
    cubeName = getAllWeatherConfig().cube_name
    if not cubeName.endswith('.'):
        cubeName = cubeName + '.'

    result = res.replace(cubeName, '')
    if result.find('date.hour') >= 0:
        # TODO: iterate TimeGranularity and change to 'time_granularity'
        # or similar if one of them is in date.<enum search>
        result = result.replace('date.hour', 'hour')

    return result
