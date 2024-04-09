import os
import torch
import pandas as pd
import pytorch_lightning as pl

# model imports
from from_pretrained import forecaster_from_pretrained
from layers.model import Forecaster
from dataset import data_module

def _predict(
        data: dict[str, pd.DataFrame],
        checkpoint_path: str,
        ) -> dict[str, pd.DataFrame]:
    """
    Returns a dict with station numbers as keys and dataframes 
    of hourly weather station predictions. Returns a 3-day 
    forecast of precipitation.
    """

    model = forecaster_from_pretrained(checkpoint_path)
    dm = data_module(data,
                     target=['precipitation']) 
    dm.setup(stage="predict")
    trainer = pl.Trainer(default_root_dir="results",
                            enable_checkpointing=False,
                            accelerator="cpu",
                            devices=1)
     
    return generate_predictions(model, trainer, dm)

def iterate_stations(preds: dict[str, pd.DataFrame],
                    result: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for s, _df in preds.items():
        result[s] = pd.concat([result[s], _df])
    return result

def generate_predictions(model: pl.LightningModule, 
                            trainer: pl.Trainer, 
                            dm: data_module, 
                            preds: dict | None = None,
                        ):
    """
    Generates 6 x 12-hour predictions
    station by station.
    """
    loader = dm.predict_dataloader
    predictions = trainer.predict(model, loader)
    result = dm.process_preds(predictions)
    for i in range(5):
        predictions = trainer.predict(model, 
                                      dm.predict_combined_loader(preds=preds))
        preds = dm.process_preds(predictions)
        for s, _df in preds.items():
            result[s] = pd.concat([result[s], _df])
    return result
