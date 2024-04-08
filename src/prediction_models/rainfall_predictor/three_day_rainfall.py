import sys
from os import path
import json

from predict import _predict
from dataset import data_module
from mutils import get_checkpoint_filepath, plot_predictions

from from_pretrained import forecaster_from_pretrained
from AllWeatherConfig import getAllWeatherConfig
from AllWeatherCubeResponse import AllWeatherCubeQueryResponse
from AllWeatherCubeResponse import cleanCubeNameFromResponseKeys

def serialise_ml_data():
    jsonData = loadJsonFixture()
    cubeName = getAllWeatherConfig().cube_name
    cleanedJson = cleanCubeNameFromResponseKeys(jsonData)
    jsonData = json.loads(cleanedJson)
    model = AllWeatherCubeQueryResponse.model_validate(jsonData)
    return model
    
def loadJsonFixture():
    '''
    load the sample Json file to the Cube query resonse model format.
    '''
    p = path.join('/home/leigh/Code/ekoh/similie/',
                    'test',
                    'fixtures', 
                    'all_weather_cube_query_response.json')
    with open(p, 'r') as file:
        jsonData = json.load(file)
        return json.dumps(jsonData)

if __name__ == "__main__":

    check_path = get_checkpoint_filepath(latent_dim=64)
    data = serialise_ml_data()
    weather_data = data.model_dump(by_alias=True)['data']
    predictions = _predict(weather_data, check_path)
    plot_predictions(predictions)
    
# pdf saved to the results folder root
