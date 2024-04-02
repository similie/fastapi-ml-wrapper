## Imports for plotting
# import matplotlib.pyplot as plt
# %matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
# from matplotlib.colors import to_rgb
# import matplotlib
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

def plot_loss(model_dict, check_pt_path):
    latent_dims = sorted([k for k in model_dict])
    # THIS NEEDS FIXING: test: test_loss/dataloader_idx
    val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]

    fig = plt.figure(figsize=(6,4))
    plt.plot(latent_dims, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Reconstruction error over latent dimensionality", fontsize=12)
    plt.xlabel("Latent dimensionality")
    plt.ylabel("Reconstruction error")
    plt.minorticks_off()
    plt.ylim(0,100)
    # plt.savefig(check_pt_path + 'loss.pdf')
    plt.show()

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
