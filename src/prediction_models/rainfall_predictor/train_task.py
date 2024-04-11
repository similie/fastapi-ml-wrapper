from train import _train
import torch

from AllWeatherConfig import AllWeatherMLConfig

"""
Supply the config from the dotEnv. Check-pointing 
handled in the _train method of the train.py file 
in file-path indicated also in the dotEnv.
"""

def trainModels():
    config = AllWeatherMLConfig() 
    model, result_ld = _train(config)
    return model, result_ld 
    
if __name__ == "__main__": 
    torch.set_default_dtype(torch.float32) # eventually add to config
    model, result = trainModels()
    print("Finished training ", model.__class__)
    del result
    # Eventually add torch.cuda.empty_cache() for GPUs 
    # but pytorch lightning is having issues...
    # https://github.com/Lightning-AI/pytorch-lightning/issues/3275
