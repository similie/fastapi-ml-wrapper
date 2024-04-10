
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

def test_datamodule():
    config = getAllWeatherConfig()
    lstm_config = config.lstm_config
    experiment_config = config.experiment_config
    trainer_config = config.trainer_config

    weather_data = serialise_ml_data()
    dm = data_module(data=weather_data)
    dm.setup(stage='fit')
    train_loader = dm.train_dataloader
    val_loader = dm.val_dataloader
    test_loader = dm.test_dataloader

    tr_it = iter(train_loader)
    ts_it = iter(test_loader)
    vl_it = iter(val_loader)
    for it in [tr_it, ts_it, vl_it]:
        batch = next(it)
        if len(batch) == 2:
            inputs, target = batch
            assert inputs.size() == torch.size(())
        else:
            inputs, target = batch[0][0], batch[0][1]
            assert inputs.size() == torch.size(())

