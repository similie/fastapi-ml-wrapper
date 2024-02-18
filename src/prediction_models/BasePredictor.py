# A base class for prediction models. Write your subclass such that it is self
# contained and in it's own folder. The class name you use should be registered
# in /src/AllModelsService->_allModels with a public 'key' that will get
# published via the /models endpoint. Subsequent calls to an endpoint
# containing a model name will attempt to instantiate a class from the value
# i.e. the Class name assocaited with that key.

import abc
from typing import Any
from ..interfaces.ReqRes import BasePostRequest


class BasePredictor(abc.ABC):
  '''
  TODO
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
  def train(self, payload: Any) -> Any:
    pass

  @abc.abstractmethod
  def fineTune(self, payload: Any) -> Any:
    pass
