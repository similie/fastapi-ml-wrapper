from pydantic import BaseModel
from .CubeJsQuery import CubeQuery


class CubeQueryRequest(BaseModel):
  '''
  Wrapper class that puts a 'query' field at the head of CubeQuery so
  that it will load from the Cube rest-api with ?query={URL encoded query}
  See: https://cube.dev/docs/product/apis-integrations/rest-api/query-format
  '''
  query: CubeQuery
