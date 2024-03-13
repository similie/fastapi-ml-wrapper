from os import path, getcwd
import math
from datetime import timedelta
from pytz import UTC
import torch
import pandas as pd
from ..AllWeatherConfig import getAllWeatherMLConfig
from ..PredictionPostResponse import (
    RainfallPrediction,
    StationRainfallPrediction,
)
from .experiment import Experiment
from .ml_models import TabulaRasa


def predict(
        startDateUTC: pd.Timestamp,
        data: pd.DataFrame,
        predictTimeOffsetDays: int = 3
    ):
    config = getAllWeatherMLConfig()
    checkPointPath = path.join(getcwd(), config.checkpoint_path, 'tabularasa_checkpoint.ckpt')
    modelWeightsPath = path.join(getcwd(), config.checkpoint_path, 'tabularasa_weights.pt')

    dateCounter = (startDateUTC + timedelta(days=predictTimeOffsetDays)).timestamp()
    
    model = TabulaRasa(**config.lstm_config.model_dump())
    model.to(torch.double)

    expt = Experiment.load_from_checkpoint(checkPointPath, model=model, check_path=modelWeightsPath)
    expt.setJsonData(data)
    expt.to(torch.double)
    expt.model.eval()
    expt.dm.make_stage(stage='predict')  # TODO: enum of possible options, use [set] on input method
    result: list[StationRainfallPrediction] = []

    with torch.no_grad():
        for s, dd in zip(expt.dm.stations, expt.dm.pred):
            predictions: list[RainfallPrediction] = []
            modelValueAccumulator = []

            for dl in dd:
                out = expt.model.forward(dl[0])
                out = expt.dm.yscaler.inverse_transform(out[-1,:])
                modelValueAccumulator.extend(out[:,0])

            for value in modelValueAccumulator:                
                predictions.append(RainfallPrediction(
                    timestamp=dateCounter,
                    precipitation=0.0 if math.isnan(value) else value
                ))
                dateCounter += 3600  # 1 hour

            result.append(StationRainfallPrediction(
                station=s,
                data=predictions
            ))

    return result
