from fastapi import BackgroundTasks
from pydantic import BaseModel
from .BasePredictor import BasePredictor
from ..interfaces.ReqRes import (
    TemplateResponse,
    BasePostRequest
)


class ATestTemplateProps(BaseModel):
    '''
    Properties model for ATestPredictor Template
    '''
    fieldOne: str = 'f1'
    fieldTwo: int = 1


class ATestPredictor(BasePredictor):
    '''
    Implementation class of abstract BasePredictor for testing
    '''
    async def template(self) -> TemplateResponse:
        t = await super().template()
        t.notes = 'test predictor schema'
        t.events = ['onTest']
        t.accepts = ATestTemplateProps.model_json_schema()
        return t

    async def predict(self, payload: BasePostRequest, taskManager: BackgroundTasks | None = None):
        await super().predict(payload)
        result = {
            'result': True,
            'count': 0,
            'payload': payload
        }
        return result
