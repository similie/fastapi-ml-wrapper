from fastapi.testclient import TestClient
from app.main import app
from src.interfaces.ReqRes import ListTypeResponse

client = TestClient(app)


def test_get_root():
    res = client.get('/api/v1')
    assert res.status_code == 200


def test_get_root_json():
    res = client.get('/api/v1')
    assert res.status_code == 200
    assert res.json() == {"msg": "Welcome to the Web ML Wrapper"}


def test_get_all_models():
    res = client.get('/api/v1/models')
    assert res.status_code == 200
    # model_validate throws if not valid, otherwise returns dict-like class
    responseData = ListTypeResponse.model_validate(res.json())
    assert responseData is not None
    assert len(responseData.data) > 0
    assert responseData.data.__contains__('test')


def test_get_all_model_contents():
    res = client.get('/api/v1/models')
    models = ListTypeResponse.model_validate(res.json())
    assert len(models.data) > 0
    assert models.data.__contains__('test')


# Section to test [40x] responses from an unknown model name
def test_get_unknown_model_template():
    res = client.get('/api/v1/template/unknown')
    assert res.status_code == 404


def test_get_unknown_model_prediction():
    '''
    Get method not allowed in /predict endpoints
    '''
    res = client.get('/api/v1/predict/unknown')
    assert res.status_code == 405


def test_post_unknown_model_prediction():
    res = client.post('/api/v1/predict/unknown', json={"foo": "bar"})
    assert res.status_code == 404


def test_get_unknown_model_fine_tune():
    '''
    Get method not allowed in /fine-tune endpoints
    '''
    res = client.get('/api/v1/fine-tune/unknown')
    assert res.status_code == 405


def test_post_unknown_model_fine_tune():
    res = client.post('/api/v1/fine-tune/unknown', json={"foo": "bar"})
    assert res.status_code == 404


def test_get_unknown_model_fine_tune_csv():
    '''
    Get method not allowed in csv upload endpoints
    '''
    res = client.get('/api/v1/fine-tune-with-csv/unknown')
    assert res.status_code == 405


def test_post_unknown_model_fine_tune_csv():
    res = client.post('/api/v1/fine-tune-with-csv/unknown', json={"foo": "bar"})
    assert res.status_code == 404


# Section to test model repsonses to the test model
def test_get_model_template():
    res = client.get('/api/v1/template/test')
    assert res.status_code == 200


def test_post_model_predict():
    res = client.post('/api/v1/predict/test', json={})
    assert res.status_code == 200


def test_post_model_fine_tune():
    res = client.post('/api/v1/fine-tune/test', json={})
    assert res.status_code == 200


def test_post_model_fine_tune_csv():
    res = client.post('/api/v1/fine-tune-with-csv/test', json={})
    assert res.status_code == 200
