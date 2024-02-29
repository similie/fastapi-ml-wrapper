import abc
from typing import Any
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
    webhooks: list[WebhookRequest]
    payload: BasePostRequest

    def __init__(self, payload) -> None:
        self.webhooks = []
        self.payload = payload

    def setWebhook(self, value: WebhookRequest):
        # TODO: check url and event name for uniqueness before adding
        self.webhooks.append(value)

    async def sendWebhook(self, req: WebhookRequest, res: WebhookResponse):
        if req.callbackUrl is not None:
            url = req.callbackUrl
            url = url + ''
            # TODO: actual POST request
            # TODO: store responses and remove webhook after [some - 3,5?] 500 response errors

    async def sendWebhookIfNeeded(self, forEventNamed: str, res: WebhookResponse):
        # TODO: manage '*' event names
        for webhook in self.webhooks:
            for eventName in webhook.eventNames:
                if eventName == forEventNamed:
                    await self.sendWebhook(webhook, res)

    async def template(self) -> TemplateResponse:
        result = TemplateResponse(
            name=self.__class__.__name__,
            accepts={},
            events=[],
            returns=BackgroundTaskResponse.model_json_schema()
            )
        return result

    @abc.abstractmethod
    async def predict(self, payload: Any) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    async def train(self, payload: Any) -> Any:
        pass

    @abc.abstractmethod
    async def fineTune(self, payload: Any) -> Any:
        pass
