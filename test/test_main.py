from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_root():
  res = client.get('/api/v1')
  assert res.status_code == 200


def test_get_root_json():
  res = client.get('/api/v1')
  assert res.json() == {"msg": "Welcome to the Web ML Wrapper"}
