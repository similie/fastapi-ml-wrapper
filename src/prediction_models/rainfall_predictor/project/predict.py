from os import path, getcwd
from datetime import datetime, timedelta
from pytz import UTC
import torch
import pandas as pd
from ..AllWeatherConfig import getAllWeatherMLConfig
from .experiment import Experiment
from .ml_models import TabulaRasa


def predict(
        startDateUTC: pd.Timestamp,
        checkPointPath: str,
        modelWeightsPath: str,
        data: pd.DataFrame,
        predictTimeOffsetDays: int = 3
    ):
    dateCounter = (startDateUTC + timedelta(days=predictTimeOffsetDays)).timestamp()
    config = getAllWeatherMLConfig()
    
    model = TabulaRasa(**config.lstm_config.model_dump())
    model.to(torch.double)

    expt = Experiment.load_from_checkpoint(checkPointPath, model=model, check_path=modelWeightsPath)
    expt.setJsonData(data)
    expt.to(torch.double)
    expt.model.eval()
    expt.dm.make_stage(stage='predict')  # TODO: enum of possible options, use [set] on input method
    result = []

    with torch.no_grad():
        for s, dd in zip(expt.dm.stations, expt.dm.pred):
            predictions = []
            p2 = []

            for dl in dd:
                out = expt.model.forward(dl[0])
                out = expt.dm.yscaler.inverse_transform(out[-1,:])
                p2.extend(out[:,0])

            # TODO: need models for correct validation and output parsing
            for p in p2:
                predictions.append({
                    'timestamp': dateCounter,
                    'precipitation': p
                })
                dateCounter += 3600  # 1 hour

            result.append({
                'station': s,
                'data': predictions
            })

    return result

def testPredict():
    basePath = 'src/prediction_models/rainfall_predictor/project/checkpoints/'
    checkPointPath = path.join(getcwd(), basePath, 'tabularasa_checkpoint.ckpt')
    modelWeightsPath = path.join(getcwd(), basePath, 'tabularasa_weights.pt')
    jsonPath = path.join(getcwd(), basePath, 'test_cube.json')
    data = pd.read_json(jsonPath)
    firstDataPoint = data["date"][0]

    print(getAllWeatherMLConfig().model_dump())


    predictions = predict(
        firstDataPoint,
        checkPointPath,
        modelWeightsPath,
        data
    )

    print(f'test predictions done: {predictions[0]['data'][0]}')
    return predictions
