from datetime import datetime, date
from pydantic import Field
from ...interfaces.ReqRes import BasePostRequest, WebhookRequest
from .AllWeatherCubeResponse import AllWeatherQueryMeasuresResponse


class CubePredictionPostRequest(BasePostRequest):
    '''
    POST Request model representing an inference request from measured station
    data that has been pre-aggregated by the Cube rollup service.
    '''
    dateRange: list[date | datetime] = Field(min_length=2, max_length=2)
    timezone: str = 'UTC'
    limit: int | None = 1000
    offset: int | None = 0
    stations: list[int] | None = []
    webhook: WebhookRequest | None = None


class ForecastPredictionPostRequest(BasePostRequest):
    '''
    POST Request model representing an inference request from a weather
    forecast. Data class contains an optional webhook and a list of:
    [weather parameters, station, datetime & hour] objects
    '''
    data: list[AllWeatherQueryMeasuresResponse]
    webhook: WebhookRequest | None = None
