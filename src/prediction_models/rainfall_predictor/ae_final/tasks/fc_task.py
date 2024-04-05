import numpy as np
import torch
from mutils import plot_loss
from fc_train import _train
from dataset import data_module

dm = data_module(target=['precipitation'])

model_dict = {}
for latent_dim in [64, 128, 256]: # , 128, 256, 512, 1024]:
    model_ld, result_ld = _train(latent_dim, dm)
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}

# plot_loss(model_dict, check_point_metrics_path)

