from os import path, listdir
import json
## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import seaborn as sns
from time import time

from layers.model import Forecaster, Autoencoder
import pytorch_lightning as pl

import torch
import pandas as pd

from AllWeatherConfig import getAllWeatherConfig
from AllWeatherCubeResponse import AllWeatherCubeQueryResponse
from AllWeatherCubeResponse import cleanCubeNameFromResponseKeys

def forecaster_from_pretrained(latent_dim: int, checkpoint_path: str) -> pl.LightningModule:

    ae_pretrain = get_pretrain_filepath(model_prefix='AE',
        latent_dim=latent_dim)
    model = Forecaster.load_from_checkpoint(checkpoint_path, 
        ae_checkpoint_path=ae_pretrain)
    model.freeze()
    return model

def autoencoder_from_pretrained(latent_dim: int, checkpoint_path: str = None) -> pl.LightningModule:
    if checkpoint_path is None:
        ae_pretrain = get_pretrain_filepath(model_prefix='AE',
            latent_dim=latent_dim)
        model = Autoencoder.load_from_checkpoint(ae_pretrain)
        model.freeze()
    else: 
        model = Autoencoder.load_from_checkpoint(checkpoint_path)
        model.freeze()
    return model

def serialise_ml_data():
    jsonData = loadJsonFixture()
    cubeName = getAllWeatherConfig().cube_name
    cleanedJson = cleanCubeNameFromResponseKeys(jsonData)
    jsonData = json.loads(cleanedJson)
    model = AllWeatherCubeQueryResponse.model_validate(jsonData)
    return model.model_dump(by_alias=True)['data']
    
def loadJsonFixture():
    '''
    load the sample Json file to the Cube query resonse model format.
    '''
    p = path.join('test',
        'test_data', 
        'all_weather_cube_query_response.json')
    with open(p, 'r') as file:
        jsonData = json.load(file)
        return json.dumps(jsonData)

def get_checkpoint_filepath(model_prefix: str = "FC",
                            latent_dim: int = 128, 
                            checkpoint_path: str ="results"):
    prefixes = ["FC", "AE"]
    if model_prefix not in prefixes:
        raise ValueError("Invalid prefix. Expected one of: %s" % prefixes)
    project_root = path.dirname(__file__)
    filename = listdir(path.join(project_root, 
        checkpoint_path,
        f"{model_prefix}_model{latent_dim}",
        "version_0/checkpoints/"))[0]
    
    return path.join(project_root, 
        checkpoint_path,
        f"{model_prefix}_model{latent_dim}",
        "version_0/checkpoints",
        filename)


def get_metrics_filepath(model_prefix: str = "FC",
                            latent_dim: int = 128, 
                            checkpoint_path: str ="results"):
    prefixes = ["FC", "AE"]
    if model_prefix not in prefixes:
        raise ValueError("Invalid prefix. Expected one of: %s" % prefixes)
    project_root = path.dirname(__file__)
    
    return path.join(project_root, 
        checkpoint_path,
        f"{model_prefix}_model{latent_dim}",
        "version_0/metrics.csv")


def get_pretrain_filepath(model_prefix: str = "FC",
                            latent_dim: int = 64, 
                            checkpoint_path: str ="pretrained_checkpoints"):
    prefixes = ["FC", "AE"]
    if model_prefix not in prefixes:
        raise ValueError("Invalid prefix. Expected one of: %s" % prefixes)
    project_root = path.dirname(__file__)
    
    return path.join(project_root, 
        checkpoint_path,
        f"{model_prefix}_model{latent_dim}",
        "pretrained.ckpt")

def plot_latent(data: pd.DataFrame, 
    ae_output: pd.DataFrame):
    ae_output.rename({'precipitation': 'ae_precip',
                        'pressure': 'ae_pressure',
                        'temperature': 'ae_temp',
                        'humidity': 'ae_humid',
                        'wind_speed': 'ae_wind',
                        'wind_direction': 'ae_wind_dir',
                        'solar': 'ae_solar'}, inplace=True)
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(12,6)})
    fig, axs = plt.subplots(2)
    fig.suptitle('Scaled Features and AE outputs prior to Rainfall Event')
    ae_output.plot(ax=axs[0])
    data.plot(ax=axs[1])

    axs[0].set_ylabel('AE predictions')
    axs[1].set_ylabel('features-scaled')
    axs[0].legend(bbox_to_anchor=(0, .5), loc='center left')
    axs[1].legend(bbox_to_anchor=(0, .5), loc='center left')
    filename = f"./plots/latent{int(time())}.pdf"
    plt.savefig(filename)

def plot_loss(metrics_dict: dict[str, any] | None = None,
    metrics_path: str | None = None):
    if metrics_dict is None:
        df = pd.read_csv(metrics_path)
    else:
        df = pd.DataFrame(metrics_dict)
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(14,8)})
    ax = sns.lineplot(data=df,
                      x = df.step % 100,
                      y = df.iloc[:,2],
                      legend='full',
                      lw=3)
    filename = f"./plots/loss{int(time())}.pdf"
    plt.savefig(filename)
    # plt.show()

def plot_predictions(preds: dict, og_data, target: str = "precipitation"):
    """
        Plot predictions generated by models
        for all the stations. Saves plot as 
        pdf in the results file.
    """
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(14,8)})
    og_key = list(og_data.keys())[0]
    og_df = og_data[og_key]
    og_df['station'] = og_key
    station_key = list(preds.keys())[0]
    _df = preds[station_key]
    _df['station'] = station_key
    target_max = _df[target].max()
    _df.rename(columns={'precipitation': 'prediction'}, inplace=True)
    _df['prediction'].plot()
    og_df['precipitation'].plot()

    # ax = sns.lineplot(data=_df,
    #                   x =_df.index.strftime('%B %d, %r'), 
    #                   y = _df[target],
    #                   legend='full', 
    #                   lw=3)
    # ax = sns.lineplot(data=og_df,
    #                   x = og_df.index.strftime('%B %d, %r'), 
    #                   y = og_df[target],
    #                   legend='full',
    #                   lw=3) 
    # ax.xaxis.set_major_locator(ticker.AutoLocator())
    # ax.set(ylim=(0, target_max))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.ylabel('rainfall (cm)')
    plt.xlabel('hour-day-month-year')
    filename = f"./plots/predictions{int(time())}.pdf"
    plt.savefig(filename)
    # plt.show()
    
def reconstruction_predictions(model, input_data):
    """ 
        Not used as of yet. Look at generate_predictions
        in predict.py TODO add this to use the AE as a 
        data imputer at some stage.
        Inputs: 
            - trained AE model
            - input data (scaled tensors)        
        Reconstruct timeseries - shifted by 12 hours
        Can feed the predictions back into the model
        to get 24 hours, etc...
    """
    model.eval()
    with torch.no_grad():
        reconst_timeseries = model(input_data.to(model.device))
    reconst_timeseries = reconst_timeseries.cpu()
    return reconst_timeseries.numpy()

