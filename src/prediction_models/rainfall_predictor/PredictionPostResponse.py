from ...interfaces.ReqRes import DataTaskResponse
from .AllWeatherCubeResponse import AllWeatherQueryMeasuresResponse


class ForecastPredictionPostResponse(DataTaskResponse):
    '''
    Override of `DataTaskResponse` narrowing the array data type to a
    list of `AllWeatherQueryMeasuresResponse`
    '''
    data: list[AllWeatherQueryMeasuresResponse]
