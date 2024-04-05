import numpy as np
import torch
from mutils import plot_ae_loss
from train import _train
from dataset import LaggedData

dp = '/home/leigh/Code/ekoh/tabula_rasa/data/combined.csv'

dm = LaggedData(dp)

model_dict = {}
for latent_dim in [64, 128, 256, 512]:
    model_ld, result_ld = _train(latent_dim, dm)
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
