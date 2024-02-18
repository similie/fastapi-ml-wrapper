from datetime import datetime
from pydantic import ConfigDict
from .CubeJsQuery import QueryMeasures
from .CubeJsQueryRequest import CubeQueryRequest


class QueryMeasuresResponse(QueryMeasures):
  '''
  These are the additional fields returned from a successful response,
  in addition to those declared in QueryMeasures (or a subclass)
  '''
  # Only present if station was supplied as a query dimension
  station: int | None = None

  # Granularity field. Only present if using date granularity and will be named
  # after the granularity used. See [interfaces.TimeGranularity] for values. E.g.
  # hour: datetime #ISO date

  date: datetime  # ISO date (always present)


class CubeQueryResponse(CubeQueryRequest):
  '''
  In general the response object will reflect the originating request. Hence
  the exact fields will vary in the dict objects contained in:
  refreshKeyValues, usedPreAggregations, transformedQuery, annotations etc.
  For brevity, we include the original query object, data list and definitely
  typed properties. The remainder are covered with ConfigDict(extra='allow')
  '''
  model_config = ConfigDict(extra='allow')
  # Inherited: query
  data: list[QueryMeasuresResponse]  # override in a subclass
  lastRefreshTime: datetime  # ISO formatted date
#   refreshKeyValues: any
#   usedPreAggregations: any
#   transformedQuery: any
  requestId: str  # UUID-ish e.g. "f3c558e7-c3b5-4cb9-adcd-2c28e19aa551-span-1"
#   annotation: dict
  dataSource: str
  dbType: str
  extDbType: str
  external: bool
  slowQuery: bool
  total: int | None
