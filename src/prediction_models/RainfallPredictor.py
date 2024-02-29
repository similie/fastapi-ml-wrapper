from typing import Any
from pydantic import ValidationError
from .BasePredictor import BasePredictor
from .rainfall_predictor.DataLoader import loadJson
from .rainfall_predictor.PredictionPostRequests import CubePredictionPostRequest, ForecastPredictionPostRequest
from .rainfall_predictor.AllWeatherCubeResponse import AllWeatherQueryMeasuresResponse


class RainfallPredictor(BasePredictor):
    '''
    Implementation class of abstract BasePredictor with typed Payloads
    '''
    def __init__(self, payload) -> None:
        super().__init__(payload)

    async def template(self):
        t = await super().template()
        t.notes = '''
        Input properties are weather data, post requests return immediately
         and call back results via the supplied webhook request
        '''

        return t

    async def __loadCubeJson(self, payload: CubePredictionPostRequest):
        stations = payload.stations if payload.stations is not None else []
        # TODO: other params: limit, skip TZ
        json = loadJson(payload.dateRange, stations)
        return json

    async def fineTune(self, payload: Any):
        return super().fineTune(payload)

    def guardPredictionPayload(
            self,
            payload: CubePredictionPostRequest | ForecastPredictionPostRequest
            ) -> CubePredictionPostRequest | ForecastPredictionPostRequest:
        '''
        Attempts to validate the payload in the Predict pathway to one of the
        expected types. Returns a validated model instance if payload can be
        validated against either class or throws.
        '''
        noop = 0
        try:
            model = CubePredictionPostRequest.model_validate(payload) # noqa F841
            return model
        except ValidationError:
            noop = noop + 0

        try:
            model = ForecastPredictionPostRequest.model_validate(payload)
            return model
        except ValidationError:
            raise

    async def predict(
            self,
            payload: CubePredictionPostRequest | ForecastPredictionPostRequest
            ) -> dict[str, Any]:
        # TODO: Concrete return type once we know the shape
        # TODO: Base class "appendIfRequired" for webhook (if both url & event are not already in the webhooks array)
        # TODO: webhook call method on super with event name
        model = self.guardPredictionPayload(payload)
        if payload.webhook is not None:
            self.webhooks.append(payload.webhook)

        data: list[AllWeatherQueryMeasuresResponse] = []
        if isinstance(model, CubePredictionPostRequest):
            cubeResult = await self.__loadCubeJson(payload)
            data.append(cubeResult.data)
        else:
            data.append(payload.data)

        # reqData is either the input weather forcast or station measurements
        # pass [data] into model for inference
        self.sendWebhookIfNeeded('*', {})
        return {
            'count': data.count,
            'message': 'from RainfallPredictor.predict',
            'input_class': payload.__class__.__name__
        }

    async def train(self, payload: Any):
        return super().train(payload)
