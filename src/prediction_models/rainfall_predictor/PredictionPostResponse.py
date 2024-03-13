from pydantic import BaseModel
from ...interfaces.ReqRes import DataTaskResponse


class RainfallPrediction(BaseModel):
    '''
    Rainfall predictions from ML model, consists of a UTC timestamp in UNIX
    millis and the predicted rainfall amount.
    '''
    timestamp: int
    precipitation: float


class StationRainfallPrediction(BaseModel):
    '''
    Wraps a sequence of date stamped rainfall predictions with their station Id
    '''
    station: int
    data: list[RainfallPrediction]


class ForecastPredictionPostResponse(DataTaskResponse):
    '''
    Override of `DataTaskResponse` narrowing the array data type to a
    list of `StationRainfallPrediction`
    '''
    data: list[StationRainfallPrediction]
