from typing import Any
from fastapi import BackgroundTasks
from pydantic import ValidationError
from .BasePredictor import BasePredictor
from ..interfaces.ReqRes import (
    BackgroundTaskResponse,
    TemplateResponse,
    WebhookRequest,
    WebhookResponse
)
from .rainfall_predictor.DataLoader import loadJson
from .rainfall_predictor.PredictionPostRequests import CubePredictionPostRequest, ForecastPredictionPostRequest
from .rainfall_predictor.AllWeatherCubeRequest import AllWeatherQueryMeasures
from .rainfall_predictor.AllWeatherCubeResponse import AllWeatherQueryMeasuresResponse


class RainfallPredictor(BasePredictor):
    '''
    Implementation class of abstract BasePredictor with typed Payloads
    '''
    def __init__(self, payload) -> None:
        super().__init__(payload)

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

        # print('in Rainfall predictor, guardPayload')
        # print(payload.model_dump_json(indent=4))
        # print(f'keys: {payload.model_dump().keys()}')
        modelDict = payload.model_dump()
        # print(f'{modelDict['modelName']}')
        # print(f'{modelDict['webhook']}')

        # hook = WebhookRequest.model_validate(modelDict['webhook'])
        # print(hook.model_dump())
        # s = '{"station":27,"avg_dew_point":8.69,"hour":"2024-02-29T13:00:00Z","date":"2024-02-29T13:00:00Z","avg_wind_direction":179.81,"avg_wind_speed":6.19,"avg_soil_moisture":null,"avg_solar":null,"avg_temperature":9.38,"avg_humidity":95,"avg_pressure":1007.38,"sum_precipitation":0.32}'
        # qms = AllWeatherQueryMeasuresResponse.model_validate_json(s)
        # print(qms.model_dump())

        try:
            model = CubePredictionPostRequest.model_validate(modelDict) # noqa F841
            return model
        except ValidationError:
            noop = noop + 0

        try:
            model = ForecastPredictionPostRequest.model_validate(modelDict)
            return model
        except ValidationError as exception:
            # print(exception.errors()[0]['type'])
            raise

    async def predict(
            self,
            payload: CubePredictionPostRequest | ForecastPredictionPostRequest,
            taskManager: BackgroundTasks | None
            ) -> BackgroundTaskResponse:
        payloadModel = self.guardPredictionPayload(payload)

        # TODO branch here for inference inline or in background if webhook

        hasWebhooks = False
        if payloadModel.webhook is not None:
            hasWebhooks = True
            self.setWebhook(payloadModel.webhook)

        data: list[AllWeatherQueryMeasuresResponse]
        if isinstance(payloadModel, CubePredictionPostRequest):
            cubeResult = await self.__loadCubeJson(payloadModel)
            data = cubeResult.data
        else:
            data = payloadModel.data

        # reqData is either the input weather forcast or station measurements
        # pass [data] into model for inference
        # await responseData = await mlModel->predict routine & fire webhook
        if hasWebhooks is True:
            responseData = [1,2,3,4,5,6]
            response = WebhookResponse(
                status=200,  # OR Status from predict method
                message=f'{payloadModel.webhook.id}',
                eventName='onPredict',
                data=responseData
            )
            taskManager.add_task(self.sendWebhookIfNeeded, response)
            # statusCode = await self.sendWebhookIfNeeded(response)
            # if statusCode != 200:
            #     # TODO: remove failed callbacks with non 200 codes after [x] fails
            #     pass

        # TODO: we need the webhookId, need to sort out the rest of the return values.
        return BackgroundTaskResponse(
            message=f'src:{payload.__class__.__name__},id:{payloadModel.webhook.id},n:{len(data)}'
        )
        #     'webhook_id': payloadModel.webhook.id if hasWebhooks else '',
        #     'count': data.count,
        #     'message': 'from RainfallPredictor.predict',
        #     'input_class': payload.__class__.__name__
        # }

    async def train(self, payload: Any):
        return super().train(payload)
