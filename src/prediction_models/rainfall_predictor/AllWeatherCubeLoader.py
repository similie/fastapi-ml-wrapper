from httpx import get, URL
# TODO: import when headers is running, Request, Headers
from urllib import parse
from .AllWeatherCubeRequest import makeAllWeatherQueryReq
from .AllWeatherCubeResponse import AllWeatherCubeQueryResponse, cleanCubeNameFromResponseKeys
from .AllWeatherConfig import getAllWeatherConfig


def loadJson(dateRange: list[str], stationIds: list[int] = []):
    config = getAllWeatherConfig()
    baseUrl = config.cube_rest_api
    req = makeAllWeatherQueryReq(dateRange, stationIds)
    modelDump = req.model_dump(mode='json', exclude_unset=True, exclude_none=True)
    # Note. Doesn't like my double quotes, switch them to %22 from %27
    queryParam = parse.urlencode(modelDump, encoding='utf-8').replace('%27', '%22')

    # TODO: Get auth token for production CubeJs instance
    # headers = Headers(['Authorization','Bearer My-Madeup-Token'])

    url = URL(baseUrl)
    res = get(url, params=queryParam)  # possibly POST to loose the URL encoding part. TODO: TEST
    status = res.status_code
    jsonString = cleanCubeNameFromResponseKeys(res.text)
    result = AllWeatherCubeQueryResponse.model_validate_json(jsonString)
    result.total = len(result.data)

    print(f"Status: {status}, rows:{result.total}, Bytes: {res.num_bytes_downloaded}")
    return result

# TODO load data frames from PG instance.
