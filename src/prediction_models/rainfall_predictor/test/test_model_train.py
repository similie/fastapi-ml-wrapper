from ..layers.model import Autoencoder, Forecaster 
from ..dataset import data_module
from ..preprocessor import load_data_csv
from ..mutils import get_checkpoint_filepath
# PyTorch Lightning
import pytorch_lightning as pl
import torch

from get_test_data import serialise_ml_data

from ...AllWeatherConfig import getAllWeatherConfig
from ...AllWeatherCubeResponse import AllWeatherCubeQueryResponse
from ...AllWeatherCubeResponse import cleanCubeNameFromResponseKeys

def test_model_init(prefix_str: str = "AE", 
    latent_dim: int = 64) -> tuple[Autoencoder, dict[str, str]]:
    """
        Test model inits.
    """

    CHECKPOINTPATH = '../pretrained_checkpoints/'
    prefixes = ["FC", "AE"]
    if prefix_str not in prefixes:
        raise ValueError("Invalid prefix. Expected one of: %s" % prefixes)
    if prefix_str == "AE":
        dm = data_module(data=weather_data)
    elif prefix_str == "FC":
        dm = data_module(data=weather_data, 
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
    return model
    
if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    weather_data = serialise_ml_data()
    model = test_model_init(prefix="AE", latent_dim=64)
    dm = data_module(weather_data)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader
    test_loader = dm.test_dataloader
    val_loader = dm.val_dataloader
    tr_it = iter(train_loader)
    ts_it = iter(test_loader)
    vl_it = iter(val_loader)
    for it in [tr_it, ts_it, vl_it]:
        batch = next(it)
        if len(batch) == 2:
            inputs, target = batch
            outputs = model.forward(inputs)
            print("Nan found in model output: ", 
                (torch.isnan(outputs).any()))
        else:
            inputs, target = batch[0][0], batch[0][1]
            outputs = model.forward(inputs)
            print("Nan found in model output:",
                (torch.isnan(outputs).any()))

