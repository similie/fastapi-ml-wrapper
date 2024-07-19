import abc
from typing import Any
from fastapi import BackgroundTasks, status
from httpx import AsyncClient, post
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

        print(f'SendWebhook. Req:{req.model_dump()}')
        print(f'SendWebhook. Res:{res.model_dump()}')
        try:
            if req.callbackUrl is not None:
                url = str(req.callbackUrl)
                jsonBody = res.model_dump_json()
                # async with AsyncClient() as client:
                headers = {
                    'Content-Type': 'application/json',
                    'X-Webhook-Token': f'{req.callbackAuthToken}'
                }
                response = post(url, json=jsonBody, headers=headers)

                #    response = await client.post(url, json=jsonBody, headers=headers)
                result = response.status_code

        except ValueError as e:  # noqa E722 TODO: find correct exception class
            print(e)
            result = 501

        return result

    async def sendWebhookIfNeeded(self, res: WebhookResponse) -> int:
        '''
        Try to match the event name to one of the listed webhooks in self & send
        it if we have one or more webhooks registered and one of them has an
        event that matches that in the specified `Webhookresponse`. If a webhook
        was found, send (post) it and return the status code of that call,
        otherwise return an information 202-accepted status code.
        '''
        # print(f'sendWebhookIfNeeded webhooks: {self.webhooks}')
        statusCode = status.HTTP_202_ACCEPTED
        hookToSend: WebhookRequest | None = None
        if self.webhooks is not None and len(self.webhooks) > 0:
            forEventNamed = res.eventName
            for webhook in self.webhooks:
                if webhook.eventNames.count(forEventNamed) > 0:
                    hookToSend = webhook

        if hookToSend is not None:
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

    # TODO: Train and Fine-tune pathways using websockets or callbacks+events
    # @abc.abstractmethod
    # async def train(self, payload: Any) -> Any:
    #     pass

    # @abc.abstractmethod
    # async def fineTune(self, payload: Any) -> Any:
    #     pass
