from os import path
# model imports
from ..layers.model import Autoencoder, Forecaster 
from ..dataset import data_module
from ..preprocessor import load_data_csv
from ..mutils import get_checkpoint_filepath
# PyTorch Lightning
import pytorch_lightning as pl

from ...AllWeatherConfig import getAllWeatherConfig
from ...AllWeatherCubeResponse import AllWeatherCubeQueryResponse
from ...AllWeatherCubeResponse import cleanCubeNameFromResponseKeys

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


def test_model_init(prefix_str: str = "AE", 
    latent_dim: int = 64) -> tuple[Autoencoder, dict[str, str]]:
    """
        Test model inits.
    """
    prefixes = ["FC", "AE"]
    if prefix_str not in prefixes:
        raise ValueError("Invalid prefix. Expected one of: %s" % prefixes)
    data_path = './test_data/all_weather_cube_query_response.json'
    df = load_data_csv(data_path)
    if prefix_str == "AE":
        dm = data_module(data=df)
    elif prefix_str == "FC":
        dm = data_module(data=df, 
                        target=["precipitation"])
        ae_checkpoint_path = get_checkpoint_filepath("AE", 
                                                 latent_dim, 
                                                 CHECKPOINTPATH)
    else:
        print("Please provide FC or AE as a prefix string.")
        
    dm.setup(stage='fit')
    
    if prefix_str == "AE":
        model = Autoencoder(input_size=7, 
                        latent_dim=latent_dim,
                        dropout=0.7,
                        output_size=7,
                        batch_size=1,)
    elif prefix_str == "FC":
        model = Forecaster(input_size=7, 
                        latent_dim=latent_dim,
                        dropout=0.5,
                        ae_checkpoint_path=ae_checkpoint_path)
    
if __name__ == "__main__":

    data = serialise_ml_data()
    weather_data = data.model_dump(by_alias=True)['data']
