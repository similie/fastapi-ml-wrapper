# import pytest
from src.prediction_models.rainfall_predictor.AllWeatherConfig import (
    AllWeatherConfig,
    getAllWeatherConfig,
    AllWeatherMLConfig,
    getAllWeatherMLConfig
)


def test_app_config():
    config = AllWeatherConfig()
    assert config is not None


def test_app_get_config():
    config = getAllWeatherConfig()
    assert config is not None


def test_app_cube_api_settings():
    config = getAllWeatherConfig()
    assert len(config.cube_name) > 0
    assert config.cube_rest_api.startswith('http')


def test_all_weather_ml_config():
    config = AllWeatherMLConfig()
    assert config is not None


def test_all_weather_get_ml_config():
    config = getAllWeatherMLConfig()
    assert config is not None


def test_all_weather_ml_config_submodels():
    config = getAllWeatherMLConfig()
    assert config.experiment_config is not None
    assert config.lstm_config is not None
    assert config.trainer_config is not None


def test_all_weather_ml_config_settings():
    config = getAllWeatherMLConfig()
    assert config.experiment_config.batch_size == 1
    assert config.experiment_config.sequence_length == 12
    assert config.lstm_config.batch_size == 1
    assert config.lstm_config.sequence_length == 12
    # precision is e.g. 32, 64 etc.
    assert pytest.approx(config.trainer_config.precision % 2) == 0
    assert len(config.trainer_config.accelerator) > 0
