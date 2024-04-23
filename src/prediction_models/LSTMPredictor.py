from typing import Any
from fastapi import BackgroundTasks
from pydantic import BaseModel
from .BasePredictor import BasePredictor
from ..interfaces.ReqRes import TemplateResponse, DataTaskResponse
# (BackgroundTaskResponse, WebhookRequest, WebhookResponse)
from .lstm_predictor.AllWeatherCubeLoader import loadJson
from .lstm_predictor.PredictionPostRequests import (
    CubePredictionPostRequest,
    ForecastPredictionPostRequest
)
from .lstm_predictor.AllWeatherCubeRequest import AllWeatherQueryMeasures
from .lstm_predictor.AllWeatherCubeResponse import AllWeatherQueryMeasuresResponse
from .lstm_predictor.predict import predict

class LstmTemplateProps(BaseModel):
    '''
    Properties model for LstmPredictor Template
    '''
    fieldOne: str = 'f1'
    fieldTwo: int = 1


class LSTMPredictor(BasePredictor):
    '''
    Implementation class of abstract BasePredictor for testing
    '''
    async def template(self) -> TemplateResponse:
        t = await super().template()
        t.notes = 'lstm predictor schema'
        t.events = ['onTest']
        t.accepts = LstmTemplateProps.model_json_schema()
        return t

    # async def fineTune(self, payload: Any):
    #     await super().fineTune(payload)
    #     result = {
    #         'result': True,
    #         'count': 0,
    #         'payload': payload
    #     }
    #     return result

    async def predict(self, payload: Any, taskManager: BackgroundTasks | None = None):
        await super().predict(payload)
        result = {
            'result': True,
            'count': 0,
            'payload': payload
        }
        return result

    async def compute_model_error(self, payload: Any):
        await super().compute_stochastic_dropout(payload)
        result = {
            'result': True,
            'count': 0,
            'payload': payload
        }
        return result
