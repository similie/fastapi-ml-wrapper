from datetime import datetime, date
from pydantic import Field
from ...interfaces.ReqRes import BasePostRequest


class FineTunePostRequest(BasePostRequest):
    '''
    POST Request model representing a fine-tuning refresh request
    '''
    dateRange: list[date | datetime] = Field(min_length=2, max_length=2)
    timezone: str = 'UTC'
    limit: int | None = 1000
    offset: int | None = 0
    stations: list[int] | None = []
    webhook: str | None = ''
