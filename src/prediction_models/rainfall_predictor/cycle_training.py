from datetime import datetime
from .train import _train

def trainingRun(dimension: int, modelType: str, epochs: int) -> dict[str, any]:
    """
    Training loop for multiple forecasters with different latent dimensions.
    Check-pointing handled in the _train method of the train.py file.
    """
    model_ld, result_ld = _train(
        prefix_str=modelType,
        latent_dim=dimension, 
        epochs=epochs
    )
    return {"model": model_ld, "result": result_ld}


if __name__ == '__main__':
    dryRun = False
    trainingResults: dict[str, any] = {}
    max_epochs = 1
    for dimension in [64, 128, 256]:
        for modelType in ['AE', 'FC']:
            start = datetime.now()
            key = f'{modelType}{dimension}'
            print(f'Starting run for: {key}')
            if dryRun is True:
                result = {"model": key, "result": "debug"}
            else:
                result = trainingRun(dimension, modelType, max_epochs)
            trainingResults[key] = result
            end = datetime.now()
            diff = (end - start).total_seconds() * (1/60)
            print(f'Finished {key} in {diff}mins')

    print('Finished training')
    print(trainingResults)
