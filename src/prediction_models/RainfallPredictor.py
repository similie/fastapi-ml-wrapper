from typing import Any
from fastapi import BackgroundTasks
from pydantic import ValidationError
from .BasePredictor import BasePredictor
from ..interfaces.ReqRes import TemplateResponse, DataTaskResponse
# TODO: when webhooks implemented: (BackgroundTaskResponse, WebhookRequest, WebhookResponse)
from .rainfall_predictor.AllWeatherCubeLoader import loadJson
from .rainfall_predictor.PredictionPostRequests import (
    CubePredictionPostRequest,
    ForecastPredictionPostRequest
)
from .rainfall_predictor.AllWeatherCubeRequest import AllWeatherQueryMeasures
from .rainfall_predictor.AllWeatherCubeResponse import AllWeatherQueryMeasuresResponse
from .rainfall_predictor.predict import predict


class RainfallPredictor(BasePredictor):
    '''
    Implementation class of abstract BasePredictor with typed Payloads
    '''
    async def template(self) -> TemplateResponse:
        t = await super().template()
        t.notes = '''
        Input properties are a list of weather data. `POST` requests return
        immediately and results are supplied via the callback parameters
        specified in the webhook
        '''
        t.events = ['onPredict', 'onForecast']
        t.accepts = AllWeatherQueryMeasures.model_json_schema()
        return t

    async def __loadCubeJson(self, payload: CubePredictionPostRequest):
        stations = payload.stations if payload.stations is not None else []
        # TODO: other params: limit, skip TZ
        json = loadJson(payload.dateRange, stations)
        return json

    # TODO when base class is implemented
    # async def fineTune(self, payload: Any):
    #     return super().fineTune(payload)

    def guardPredictionPayload(
            self,
            payload: CubePredictionPostRequest | ForecastPredictionPostRequest
            ) -> CubePredictionPostRequest | ForecastPredictionPostRequest:
        '''
        Attempts to validate the payload in the Predict pathway to one of the
        expected types. Returns a validated model instance if payload can be
        validated against either input class or throws.
        '''
        noop = 0
        modelDict = payload.model_dump()

        try:
            model = CubePredictionPostRequest.model_validate(modelDict) # noqa F841
            return model
        except ValidationError:
            noop = noop + 0

        try:
            model = ForecastPredictionPostRequest.model_validate(modelDict)
            return model
        except ValidationError:  # as exception
            # print(exception.errors()[0]['type'])
            raise

    async def predict(
            self,
            payload: CubePredictionPostRequest | ForecastPredictionPostRequest,
            taskManager: BackgroundTasks | None
            ):
        payloadModel = self.guardPredictionPayload(payload)
        # TODO branch here for inference inline (or in background if webhook supplied)
        # For straight inference, return a ForecastPredictionPostResponse. If a webhook
        # was supplied, return a BackgroundTaskResponse and add result data to the
        # WebhookResponse instance, returned via `sendWebhookIfNeeded`

        data: list[AllWeatherQueryMeasuresResponse]
        if isinstance(payloadModel, CubePredictionPostRequest):
            cubeResult = await self.__loadCubeJson(payloadModel)
            data = cubeResult.data
        else:
            data = payloadModel.data

        # request `data` is now either the input weather forcast or station measurements
        # pass [data] into model for inference
        predictions = predict(
            data
            # TODO: check if we should pass first or last date into Predictor.
            # startDateUTC=data[0].date,
            # predictTimeOffsetDays=3  # or get from config
        )

        return DataTaskResponse(
            status=200,
            message=f'Inference in {self.__class__.__name__},count:{len(predictions[0].data)}',
            data=predictions
        )
