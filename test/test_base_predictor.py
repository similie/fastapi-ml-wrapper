from typing import Any
import pytest
from src.interfaces.ReqRes import BasePostRequest
from src.prediction_models.TestPredictor import ATestPredictor


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
    result = await predictor.template()
    assert result is not None
    assert result.keys().__contains__('schema')
    assert result['schema'] == 'test predictor schema'


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
