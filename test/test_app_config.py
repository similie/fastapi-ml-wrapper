from app.config import ConfigSettings, getConfig


def test_app_config():
    config = ConfigSettings()
    assert config is not None


def test_get_app_config():
    config = getConfig()
    assert config is not None


def test_api_endpoint():
    config = getConfig()
    assert config.apiEndpoint.startswith('/')
    assert config.apiEndpoint.startswith('/api')
