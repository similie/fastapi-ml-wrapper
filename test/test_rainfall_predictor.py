import pytest
from ..src.interfaces.ReqRes import BasePostRequest
from ..src.prediction_models.RainfallPredictor import RainfallPredictor
# from src.prediction_models.rainfall_predictor.AllWeatherCubeRequest import CubeQueryRequest


def MockPostRequest() -> BasePostRequest:
    # all_weather_cube_query_response.json (.data) contains a weather forecast fixture we can use
    payload = {'modelName': 'rainfall', 'device': '123ABC', 'numeric': 1.234}
    result = BasePostRequest.model_validate(payload)
    return result


def MockTestPredictor() -> RainfallPredictor:
    req = MockPostRequest()
    return RainfallPredictor(req)


def test_is_class_initialised():
    '''
    This tests the creation of a [xxx]PredictorClass from a payload
    containing a model_name key:value pair
    '''
    req = MockPostRequest()
    predictor = RainfallPredictor(req)
    assert predictor is not None
    assert predictor.payload.modelName == 'rainfall'
    assert predictor.__class__.__name__ == 'RainfallPredictor'
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
    assert t.name == 'RainfallPredictor'
    assert t.events.index('onPredict') >= 0

# TODO: mock both prediction pathways with their respective payloads
