from typing import Any
from .BasePredictor import BasePredictor
from .rainfall_predictor.DataLoader import loadJson


class RainfallPredictor(BasePredictor):
  '''
  Implementation class of abstract BasePredictor with typed Payloads
  '''
  def __init__(self, payload) -> None:
    super().__init__(payload)

  async def template(self) -> dict[str, Any]:
    await super().template()
    result: dict[str, Any] = {}
    result['foo'] = 'bar'

    return result

  async def __loadCubeJson(self):
    json = loadJson(['2020-03-05', '2020-03-12'])
    return json

  async def fineTune(self, payload: Any):
    return super().fineTune(payload)

  async def predict(self, payload: Any):
    return await super().predict(payload)

  async def train(self, payload: Any):
    return super().train(payload)
