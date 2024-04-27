from typing import Any
import pytest
from ..src.interfaces.ReqRes import (
    BasePostRequest,
    WebhookRequest
)
from ..src.prediction_models.TestPredictor import ATestPredictor


def MockPostRequest() -> BasePostRequest:
    payload = {'modelName': 'test', 'device': '123ABC', 'numeric': 1.234}
    result = BasePostRequest.model_validate(payload)
    return result


def MockTestPredictor() -> ATestPredictor:
    req = MockPostRequest()
    return ATestPredictor(req)


def test_is_class_initialised():
    '''
    This tests the creation of a [xxx]PredictorClass from a payload
    containing a model_name key:value pair
    '''
    req = MockPostRequest()
    predictor = ATestPredictor(req)
    assert predictor is not None
    assert predictor.payload.modelName == 'test'
    assert predictor.__class__.__name__ == 'ATestPredictor'
    del predictor


def test_mock_instance_creator():
    '''
    Test our MockPredictor is creating the correct class instance.
    '''
    predictor = MockTestPredictor()
    assert predictor is not None


@pytest.mark.asyncio
async def test_predictor_template():
    predictor = MockTestPredictor()
    t = await predictor.template()
    assert t is not None
    assert t.name == 'ATestPredictor'
    assert t.notes == 'test predictor schema'


@pytest.mark.asyncio
async def test_predictor_predict():
    predictor = MockTestPredictor()
    predictionPayload = {'f1': 0, 'f2': 1}
    result: dict[str, Any] = await predictor.predict(predictionPayload)
    assert result is not None
    assert result['count'] == 0
    assert result['payload']['f1'] is not None


@pytest.mark.asyncio
async def test_predictor_finetune():
    predictor = MockTestPredictor()
    predictionPayload = {'f1': 0, 'f2': 1}
    result: dict[str, Any] = await predictor.fineTune(predictionPayload)
    assert result is not None
    assert result['count'] == 0
    assert result['payload']['f1'] is not None


@pytest.mark.asyncio
async def test_predictor_train():
    predictor = MockTestPredictor()
    predictionPayload = {'f1': 0, 'f2': 1}
    result: dict[str, Any] = await predictor.train(predictionPayload)
    assert result is not None
    assert result['count'] == 0
    assert result['payload']['f1'] is not None


def test_add_webhook():
    predictor = MockTestPredictor()
    hook = WebhookRequest(
        modelName='test',
        callbackUrl='http://example.com/callback',
        callbackAuthToken='e572b49a-f075-4c74-9be7-f3b5eb7ed33c',
        eventNames=['onTest']
    )
    assert hook.id is not None
    assert predictor.webhooks is not None
    assert len(predictor.webhooks) == 0

    predictor.setWebhook(hook)
    assert len(predictor.webhooks) == 1
