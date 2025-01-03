from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
# from dotenv import load_dotenv
# load_dotenv()


class AllWeatherConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    # custom properties (provide defaults or environment vars)
    cube_name: str = "all_weather"
    cube_rest_api: str = "http://localhost:4000/cubejs-api/v1/load"
    cube_port: int = 4000
    cube_auth_key: str = ""


class ExperimentConfig(BaseSettings):
    '''
    Experiments config settings, with .env namespace aliases
    '''
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='experiment_')
    target_col: list[str] = ["precipitation"]
    features: list[str] = [
        "precipitation", "temperature", "humidity", "pressure", "wind_speed", "wind_direction", "solar"
        ]
    prediction_window: int = 12


class TrainerConfig(BaseSettings):
    '''
    Trainer config settings, with .env namespace aliases
    '''
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='trainer_')
    accelerator: str = "cpu"
    dtype: str = "np.float32"
    pretrained_path: str = "pretrained_checkpoints"
    num_workers: int = 1


class AllWeatherMLConfig(BaseSettings):
    '''
    Namespaced ML config settings from .env
    '''
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    experiment_config: ExperimentConfig = ExperimentConfig(BaseSettings)
    trainer_config: TrainerConfig = TrainerConfig()
    # relative path from the project root to your pretrained checkpoints.
    # Note for docs. Useful for sub-classes to add their own paths to their own checkpoints
    inference_checkpoints: str = "src/prediction_models/rainfall_predictor/pretrained_checkpoints/"


@lru_cache
def getAllWeatherConfig():
    '''
    Lazy load all weather config
    '''
    return AllWeatherConfig()


@lru_cache
def getAllWeatherMLConfig():
    '''
    Lazy loader for All weather ML models. Note. In some cases, certain values
    listed here might be overidden by input params received via the API. Check
    the README.md in rainfall_predictor for variables and their default values.
    '''
    return AllWeatherMLConfig()
