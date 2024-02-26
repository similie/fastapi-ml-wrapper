from enum import Enum
from typing import Any
from datetime import datetime, date
from pydantic import BaseModel, ConfigDict


class TimeGranularity(str, Enum):
    '''
    Enum of time aggregation granularity values from the CubeJs docs.
    See: https://cube.dev/docs/product/apis-integrations/rest-api/query-format
    '''
    Year = 'year'
    Quarter = 'quarter'
    Month = 'month'
    Week = 'week'
    Day = 'day'
    Hour = 'hour'
    Minute = 'minute'
    Second = 'second'


class QueryMeasures(BaseModel):
    '''
    Abstract class designed to be subclassed to include your specific query
    measures. This will allow auto creation of measures in your query and
    validation of the resulting query from CubeJs
    IMPORTANT: Subclasses should add the expected field names & types, with
    defaults, for their specfic cube query.
    '''
    model_config = ConfigDict(extra='allow')

    def measures(self, prefix: str = '') -> list[str]:
        '''
        Exports a list of strings in the format: cubename.fieldname that can
        be assigned to a query's measures field.
        '''
        s = prefix
        if (len(prefix) > 0) and not (prefix.endswith('.')):
            s = s + '.'

        result: list[str] = []
        for fieldName in self.model_fields.keys():
            result.append(f'{s}{fieldName}')

        return result


class TimeDimension(BaseModel):
    '''
    The date field and granularity by which the aggregation table was accumulated
    '''
    dimension: str
    granularity: TimeGranularity
    dateRange: tuple[datetime | date, datetime | date]


class QueryFilter(BaseModel):
    '''
    A filter to apply to a query. Note not normally including a date filter
    since this is applied within the TimeDimension class. Filters in the final
    query are supplied as an array (of filter objects)
    '''
    member: str
    # ENUM of operators that each depend on the type of the underlying member field
    operator: str
    values: list[int]


class CubeQuery(BaseModel):
    '''
    Query definition for CubeJs Rest-API call.
    See: https://cube.dev/docs/product/apis-integrations/rest-api/query-format
    ### Order field.
    Is a dictionary object for query requests of the form cubeName.fieldName: asc | desc:
    "order": {
      "stories.time": "asc",
      "stories.count": "desc"
    },

    The query is always returned in the response, but the order is reformatted
    into a list of {'id' cubeName.fieldName, desc: [true|false]} one entry for each object
    specfied in the request:
    "order": [
      {
        "id": "all_weather.date",
        "desc": false
      }, {...}
    ],

    The typing definition for this is encapsulated in the OR definition below
    to keep the model hierarchy simpler (& since we don't normally need the query
    on the way out anyway)
    '''
    measures: list[str]
    timeDimensions: list[TimeDimension]
    order: dict[str, str] | list[dict[str, Any]]  # see comments above
    filters: list[QueryFilter] = []
    dimensions: list[str] = []
    timezone: str = 'UTC'
    limit: int | None = 20000
    offset: int | None = 0
