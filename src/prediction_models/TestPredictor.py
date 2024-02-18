from typing import Any
from .BasePredictor import BasePredictor


class TestPredictor(BasePredictor):
  '''
  Implementation class of abstract BasePredictor for testing
  '''
  async def template(self):
    return {'schema': 'test predictor schema'}

  async def fineTune(self, payload: Any):
    return super().fineTune(payload)

  async def predict(self, payload: Any):
    return await super().predict(payload)

  async def train(self, payload: Any):
    return super().train(payload)
