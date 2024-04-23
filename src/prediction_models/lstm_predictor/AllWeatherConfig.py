from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()

class AllWeatherConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    # custom properties (provide defaults or environment vars)
    cube_name: str = 'all_weather'
    cube_rest_api: str = 'http://localhost:4000/cubejs-api/v1/load'
    cube_port: int = 4000
    cube_auth_key: str = ''

class ExperimentConfig(BaseSettings):
    '''
    Experiments config settings, with .env namespace aliases
    '''
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='experiment_')
    target_col: list[str]
    features: list[str]
    prediction_window: int = 12
    
class TrainerConfig(BaseSettings):
    '''
    Trainer config settings, with .env namespace aliases
    '''
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='trainer_')
    accelerator: str
    dtype: str
    pretrained_path: str
    num_workers: int

class AllWeatherMLConfig(BaseSettings):
    '''
    Namespaced ML config settings from .env
    '''
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    experiment_config: ExperimentConfig = ExperimentConfig()
    trainer_config: TrainerConfig = TrainerConfig()

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
