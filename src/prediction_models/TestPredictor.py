from typing import Any
from .BasePredictor import BasePredictor


class ATestPredictor(BasePredictor):
    '''
    Implementation class of abstract BasePredictor for testing
    '''
    async def template(self):
        await super().template()
        return {'schema': 'test predictor schema'}

    async def fineTune(self, payload: Any):
        await super().fineTune(payload)
        result = {
            'result': True,
            'count': 0,
            'payload': payload
        }
        return result

    async def predict(self, payload: Any):
        await super().predict(payload)
        result = {
            'result': True,
            'count': 0,
            'payload': payload
        }
        return result

    async def train(self, payload: Any):
        await super().train(payload)
        result = {
            'result': True,
            'count': 0,
            'payload': payload
        }
        return result
