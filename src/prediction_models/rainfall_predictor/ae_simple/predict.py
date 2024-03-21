import os
from os import path, getcwd
import math
import torch
from datetime import timedelta
from pytz import UTC
import pandas as pd
from ..AllWeatherConfig import getAllWeatherMLConfig
from ..PredictionPostResponse import (
    RainfallPrediction,
    StationRainfallPrediction,
)
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
# model imports
from layers.model import Autoencoder
from mutils import generate_datetime_index
from dataset import get_dm 
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
  print("Install lightning...")

# Features: (set in preprocessor.py in the load_data_json function)
# ['precipitation', 'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'solar']

CHECKPOINTPATH = os.path.join(os.getcwd(), './results')

def _predict(
        startDateUTC: pd.Timestamp,
        data: pd.DataFrame,
        predictTimeOffsetDays: int = 3,
        latent_dim = 64,
        ):
    """
        Returns a dict with station numbers as keys and dataframes 
        of hourly weather station predictions. Whatever time frame 
        you supply, the system will return 6 times the length up to 
        a limit of 12 hours, or 6 * 12 = 72 hours / 3 days.
    """
    config = getAllWeatherMLConfig()
    try:
        modelWeightsPath = os.path.join(CHECKPOINTPATH + f"/AE_model{latent_dim}/version_0/checkpoints/, os.listdir(os.path.join(CHECKPOINTPATH, f"AE_model{latent_dim}/version_0/checkpoints/"))[0])
    except FileNotFoundError:
        print("Model weights not found...")
    checkPointPath = CHECKPOINTPATH

    dateCounter = (startDateUTC + timedelta(days=predictTimeOffsetDays)).timestamp()
    dm = get_dm(data_dir=data) 
    data_loader = dm.predict_combined_loader()
    
    trainer = pl.Trainer(default_root_dir=checkPointPath,
                            enable_checkpointing=False,
                            accelerator="cpu",
                            devices=1)
    
    if os.path.isfile(modelWeightsPath):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(modelWeightsPath)
    else:
        print("Pretrained model not found...")
    # Feed predictions into dataframes with an extended _df.index
    predictions = generate_predictions(model, trainer, dm)
    return predictions    

def generate_predictions(model, trainer, dm, preds=None):
    """
        Generate 6 * 12 hours of predictions, 
        either raw, or to be fed as a latent- 
        space representation into a second
        specialized forecaster model.
    """
    result = {}
    loader = dm.predict_combined_loader(preds=preds)
    predictions = trainer.predict(model, loader)
    result = dm.process_preds(predictions)
    for i in range(5):
        predictions = trainer.predict(model, 
                                      dm.predict_combined_loader(preds=preds))
        preds = dm.process_preds(predictions)
        for s, _df in preds.items():
            result[s] = pd.concat([result[s], _df])
    return result
    
