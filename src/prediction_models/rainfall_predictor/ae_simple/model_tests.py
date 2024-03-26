import os
import torch
from layers.model import Forecaster
from verify import verify_model
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
  print("Install lightning...")

latent_dim = 64
CHECKPOINTPATH = os.path.join(os.getcwd(), './results')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


try:
    modelWeightsPath = os.path.join(CHECKPOINTPATH + f"/FC_model{latent_dim}/version_0/checkpoints/", os.listdir(os.path.join(CHECKPOINTPATH, f"FC_model{latent_dim}/version_0/checkpoints/"))[0])
except FileNotFoundError:
    print("Model weights not found...")

inputs = torch.randn(1, 12, 7)
targets = torch.randn(1, 12, 1).long()
batch = inputs

if os.path.isfile(modelWeightsPath):
    print("Found pretrained model, loading...")
    model = Forecaster.load_from_checkpoint(modelWeightsPath)
    model.freeze()
else:
    print("Pretrained model not found...")

output = verify_model(model, batch, device)
assert output.size() == targets.size()
