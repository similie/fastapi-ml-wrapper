import numpy as np
# import torch
from mutils import plot_loss
from train import _train
from dataset import get_dm

dm = get_dm()

model_dict = {}
for latent_dim in [512]: # [64, 128, 256]: #, 256, 512, 1024]:
    model_ld, result_ld = _train(latent_dim, dm)
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}

