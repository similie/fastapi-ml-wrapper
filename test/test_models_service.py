import pytest
from src.interfaces.ReqRes import BasePostRequest
from src.prediction_models.AllModelsService import getModelNames, modelForPayload


def test_get_all_model_names():
    modelNames = getModelNames()
    assert modelNames is not None
    assert len(modelNames) > 0
    assert modelNames.__contains__('test')


def test_model_for_payload():
    model = modelForPayload(BasePostRequest(modelName='test'))
    assert model is not None
    assert model.__class__.__name__ == 'ATestPredictor'


def test_model_for_unknown_value():
    model = modelForPayload(BasePostRequest(modelName='unknown'))
    assert model is None


def test_model_for_unknown_payload_key():
    with pytest.raises(Exception) as exceptionInfo:
        # noqa flag ignores unused variable rule
        model = modelForPayload({'unknown': 'test'})  # noqa: F841

    assert "Error" in str(exceptionInfo.typename)
