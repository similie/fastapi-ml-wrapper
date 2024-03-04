import abc
from typing import Any
from fastapi import BackgroundTasks
from httpx import AsyncClient
from ..interfaces.ReqRes import BasePostRequest, WebhookRequest, WebhookResponse
from ..interfaces.ReqRes import TemplateResponse, BackgroundTaskResponse


class BasePredictor(abc.ABC):
    '''
    Abstract base class for Prediction models. API controller routes will call
    an instance of a sub-class based on the model name passed to the route. The
    class name you use must be registered in /src/AllModelsService->_allModels
    with a public 'key' that will get published via the /models endpoint.
    Subsequent calls to endpoints requiring a model name will attempt to
    instantiate a class from the Class name associated with that key.
    '''
    payload: BasePostRequest
    webhooks: list[WebhookRequest]

    def __init__(self, payload) -> None:
        self.payload = payload
        self.webhooks = []

    def setWebhook(self, value: WebhookRequest):
        # TODO: check url and event name for uniqueness before adding
        self.webhooks.append(value)

    async def sendWebhook(self, req: WebhookRequest, res: WebhookResponse) -> int:
        # TODO: store responses and remove webhook after [some - 3,5?] 500 response errors
        result = 200

        try:
            if req.callbackUrl is not None:
                url = req.callbackUrl
                jsonBody = res.model_dump_json()
                async with AsyncClient() as client:
                    headers = {
                        'Content-Type': 'application/json',
                        'X-Webhook-Token': f'{req.callbackAuthToken}'
                    }
                    response = await client.post(url, json=jsonBody, headers=headers)
                    result = response.status_code

        except:  # noqa E722 TODO: find correct exception class
            result = 501

        return result

    async def sendWebhookIfNeeded(self, res: WebhookResponse) -> int:
        '''
        Try to match the event name to one of the listed webhooks in self. If
        found, send the WebhookResponse and return the status code of that call
        '''
        print(f'sendWebhookIfNeeded webhooks: {self.webhooks}')
        hookToSend: WebhookRequest | None = None
        if self.webhooks is not None and len(self.webhooks) > 0:
            forEventNamed = res.eventName
            for webhook in self.webhooks:
                if webhook.eventNames.count(forEventNamed) > 0:
                    hookToSend = webhook

        statusCode = await self.sendWebhook(hookToSend, res)
        return statusCode

    async def template(self) -> TemplateResponse:
        result = TemplateResponse(
            name=self.__class__.__name__,
            accepts={},
            events=[],
            returns=BackgroundTaskResponse.model_json_schema()
            )
        return result

    @abc.abstractmethod
    async def predict(self, payload: Any, taskManager: BackgroundTasks | None = None) -> dict[str, Any]:
        '''
        Runs inference with the selected model. If the payload contains a
        webhook, the response will be send immediately with the webhook uuid
        and inference will be executed on a background thread, finishing with
        a POST requset to the callback URL. If there is no webhook specified,
        the inference call will be awaited and returned when complete.
        '''
        pass

    @abc.abstractmethod
    async def train(self, payload: Any) -> Any:
        pass

    @abc.abstractmethod
    async def fineTune(self, payload: Any) -> Any:
        pass
