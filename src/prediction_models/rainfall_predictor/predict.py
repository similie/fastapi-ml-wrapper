import os
import torch
import pandas as pd
import pytorch_lightning as pl

# model imports
from mutils import (forecaster_from_pretrained, 
                    get_pretrain_filepath, 
                    serialise_ml_data)
from layers.model import Forecaster
from dataset import data_module
from AllWeatherConfig import getAllWeatherMLConfig


config = getAllWeatherMLConfig()
target_col = config.experiment_config.target_col
sequence_length = config.experiment_config.sequence_length

def _predict(
        data: dict[str, pd.DataFrame] | None = None,
        checkpoint_path: str | None = None,
        ) -> dict[str, pd.DataFrame]:
    """
    Supply data, checkpoint_path or data will be loaded from
    the test/test_data folder.
    Returns a dict with station numbers as keys and dataframes 
    of hourly weather station predictions--a 3-day forecast of 
    precipitation.
    """

    if checkpoint_path is None:
        pretrain_path = get_pretrain_filepath(config.lstm_config.prefix,
            config.lstm_config.latent_dim)
    if data is None:
        data = serialise_ml_data()
    model = forecaster_from_pretrained(pretrain_path)

    dm = data_module(data=data,
                     target=target_col)
    dm.setup(stage="predict")
    trainer = pl.Trainer(enable_checkpointing=False,
        accelerator="cpu",
        devices=1)
     
    return generate_predictions(model, trainer, dm)

def generate_predictions(model: pl.LightningModule, 
    trainer: pl.Trainer, 
    dm: data_module, 
    preds: dict | None = None):

    """
    Generates 6 x 12-hour predictions
    station by station.
    """
    result = {}
    for _ in range(6):
        loader = dm.predict_dataloader
        predictions = trainer.predict(model, loader)
        preds = dm.process_preds(predictions)
        for s, _df in preds.items():
            if s in result:  
                result[s] = pd.concat([result[s], _df])
            else:
                result[s] = _df
        dm = data_module(data=preds, target=target_col)
        dm.setup(stage='predict')
    return result