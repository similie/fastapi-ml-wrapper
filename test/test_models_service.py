import pytest
from src.interfaces.ReqRes import BasePostRequest
from src.prediction_models.AllModelsService import (
    getModelNames,
    modelForPayload,
    ensureValidModelName
)


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


def test_ensure_valid_model_name():
    req = BasePostRequest(modelName='test')
    payload = ensureValidModelName('test', req)
    assert payload is not None


def test_ensure_valid_model_name_2():
    req = BasePostRequest(modelName='not-valid')
    payload = ensureValidModelName('test', req)
    assert payload is not None
    assert payload.modelName == 'test'


def test_ensure_valid_model_name_3():
    req = BasePostRequest(modelName='test')
    payload = ensureValidModelName('not-valid', req)
    assert payload is not None
    assert payload.modelName == 'test'


def test_ensure_valid_model_name_4():
    req = BasePostRequest(modelName='not-valid')
    payload = ensureValidModelName('not-valid', req)
    assert payload is None
