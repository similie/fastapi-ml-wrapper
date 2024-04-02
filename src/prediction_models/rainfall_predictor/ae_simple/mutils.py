## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
# %matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
# from matplotlib.colors import to_rgb
# matplotlib.rcParams['lines.linewidth'] = 2.0

import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    x = [item[0] for item in batch]
    x = pad_sequence(x, batch_first=True)
    y = [item[1] for item in batch]
    y = pad_sequence(y, batch_first=True)
    return x, y

def plot_predictions(predictions: dict, data: dict = None):
    num_stations = len(predictions)
    colors = iter(cm.rainbow(np.linspace(0, 1, num_stations)))
    fig = plt.figure(figsize=(6,4))
    for station, _df in predictions.items():
        ax = f"ax{station}"
        ax = _df.precipitation.plot(color=next(colors), grid=True, label=station)
    plt.title("Rainfall Predictions", fontsize=12)
    plt.xlabel("Hourly 3-day Forecast")
    plt.ylabel("Rainfall (cm)")
    plt.minorticks_off()
    plt.ylim(0,10)
    plt.savefig(f"predictions.jpg")

def reconstruction_predictions(model, input_data):
    """ 
        Not used as of yet. Look at generate_predictions
        in predict.py...
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

def generate_datetime_index(start_time, periods=12):
    return pd.date_range(start=start_time, freq='h', periods=periods)
