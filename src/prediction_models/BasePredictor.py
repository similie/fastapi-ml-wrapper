import abc
from typing import Any
from ..interfaces.ReqRes import BasePostRequest


class BasePredictor(abc.ABC):
    '''
    Abstract base class for Prediction models. API controller routes will call
    an instance of a sub-class based on the model name passed to the route. The
    class name you use must be registered in /src/AllModelsService->_allModels
    with a public 'key' that will get published via the /models endpoint.
    Subsequent calls to endpoints requiring a model name will attempt to
    instantiate a class from the Class name associated with that key.
    '''
    webhooks: list[Any]
    payload: BasePostRequest

    def __init__(self, payload) -> None:
        self.webhooks = []
        self.payload = payload

    @abc.abstractmethod
    async def template(self) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    async def predict(self, payload: Any) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    async def train(self, payload: Any) -> Any:
        pass

    @abc.abstractmethod
    async def fineTune(self, payload: Any) -> Any:
        pass
