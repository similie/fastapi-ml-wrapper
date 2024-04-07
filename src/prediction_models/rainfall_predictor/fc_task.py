from train import _train
from dataset import data_module
"""
Training loop for multiple forecaters with 
different latent dimensions. Check
-pointing handled in the _train method
of the train.py file.
"""
model_dict = {}
for latent_dim in [64]: # , 128, 256, 512, 1024]:
    model_ld, result_ld = _train(prefix_str="FC",
                                latent_dim=latent_dim, 
                                epochs=1)
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}


