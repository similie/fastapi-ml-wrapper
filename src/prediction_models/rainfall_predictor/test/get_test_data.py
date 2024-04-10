from os import path
# model imports
from ..dataset import data_module
from ..preprocessor import load_data_csv
from ..mutils import get_checkpoint_filepath
# PyTorch Lightning
import pytorch_lightning as pl
import torch

from ..AllWeatherConfig import getAllWeatherConfig
from ..AllWeatherCubeResponse import AllWeatherCubeQueryResponse
from ..AllWeatherCubeResponse import cleanCubeNameFromResponseKeys

def serialise_ml_data():
    jsonData = loadJsonFixture()
    cubeName = getAllWeatherConfig().cube_name
    cleanedJson = cleanCubeNameFromResponseKeys(jsonData)
    jsonData = json.loads(cleanedJson)
    model = AllWeatherCubeQueryResponse.model_validate(jsonData)
    return model.model_dump(by_alias=True)['data']
    
def loadJsonFixture():
    '''
    load the sample Json file to the Cube query resonse model format.
    '''
    p = path.join('./',
        'test_data',
        'all_weather_cube_query_response.json')
    with open(p, 'r') as file:
        jsonData = json.load(file)
        return json.dumps(jsonData)
