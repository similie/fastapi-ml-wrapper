from os import path, getcwd
import json


def loadJsonFixture() -> str:
    '''
    load the sample Json file to the Cube query resonse model format.
    Note: The query JSON contains:
    - station Ids [27]
    - date range: ["2020-03-05T00:00:00.000", "2020-03-12T23:59:59.999"]
    '''
    p = path.join(
        getcwd(), 'test',
        'fixtures',
        'all_weather_cube_query_response.json'
    )
    with open(p, 'r') as file:
        jsonData = json.load(file)
        return json.dumps(jsonData)
