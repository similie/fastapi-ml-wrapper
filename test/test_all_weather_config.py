from src.prediction_models.rainfall_predictor.AllWeatherConfig import AllWeatherConfig, getAllWeatherConfig


def test_app_config():
    config = AllWeatherConfig()
    assert config is not None


def test_get_config():
    config = getAllWeatherConfig()
    assert config is not None


def test_api_endpoint():
    config = getAllWeatherConfig()
    assert len(config.cube_name) > 0
    assert config.cube_rest_api.startswith('http')
