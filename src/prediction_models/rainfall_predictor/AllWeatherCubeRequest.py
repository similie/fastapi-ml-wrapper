from ...interfaces.CubeJsQuery import TimeDimension, CubeQuery, QueryFilter
from ...interfaces.CubeJsQueryResponse import QueryMeasures
from ...interfaces.CubeJsQueryRequest import CubeQueryRequest
from .AllWeatherConfig import getAllWeatherConfig


class AllWeatherQueryMeasures(QueryMeasures):
  '''
  Base all-weather field names that an all-weather station aggregation
  cube is called with. These should be prefixed by the cube_name before
  submitting a query to the Cube rest-api.
  '''
  avg_wind_direction: float | None = 0
  avg_wind_speed: float | None = 0
  avg_soil_moisture: float | None = 0
  avg_dew_point: float | None = 0
  avg_solar: float | None = 0
  avg_temperature: float | None = 0
  avg_humidity: float | None = 0
  avg_pressure: float | None = 0
  sum_precipitation: float | None = 0


def makeAllWeatherQueryReq(dateRange: list[str], stationIds: list[int] = []):
  '''
  Formats the daterange, stations Id(s) and cube name into an all weather query request
  object that can be passed to a CubeJs rest-api service (via a url-encoded string)
  '''
  config = getAllWeatherConfig()
  cubeName = config.cube_name
  # 1. Add hourly time dimension
  td = TimeDimension(
    dimension=f'{cubeName}.date',
    granularity='hour',
    dateRange=dateRange)
  # 2. Add all fields from the query measures class
  measures = AllWeatherQueryMeasures().measures(prefix=cubeName)

  # 3. Make the cube query with all required fields, ensuring date order
  query = CubeQuery(
    measures=measures,
    timeDimensions=[td],
    order={f'{cubeName}.date': 'asc'}
  )

  # 4. If Station Id(s) are specified, add filters and station dimension
  if len(stationIds) > 0:
    dimensions: list[str] = []
    filters: list[QueryFilter] = []
    dimensions.append(f'{cubeName}.station')
    filters.append(QueryFilter(
      member=f'{cubeName}.station',
      operator='equals',
      values=stationIds
    ))
    query.filters = filters
    query.dimensions = dimensions

  # Result. Build and return the query request
  return CubeQueryRequest(query=query)
