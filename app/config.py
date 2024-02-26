from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    # custom properties (provide defaults or environment vars)
    app_name: str = 'FastAPI ML Web Wrapper'
    admin_email: str
    items_per_user: int = 50
    foo: str
    bar: str
    api_endpoint: str = 'api/v1'

    @property
    def apiEndpoint(self) -> str:
        return self.api_endpoint if (self.api_endpoint.startswith('/')) else f'/{self.api_endpoint}'


@lru_cache
def getConfig():
    '''
    Lazy load app config
    '''
    return ConfigSettings()
