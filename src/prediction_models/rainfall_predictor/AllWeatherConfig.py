from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class AllWeatherConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    # custom properties (provide defaults or environment vars)
    cube_name: str = 'all_weather'
    cube_rest_api: str = 'http://localhost:4000/cubejs-api/v1/load'
    cube_port: int = 4000
    cube_auth_key: str = ''


@lru_cache
def getAllWeatherConfig():
    '''
    Lazy load all weather config
    '''
    return AllWeatherConfig()
