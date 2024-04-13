import sys
from os import path
import json

from predict import _predict
from dataset import data_module
from mutils import plot_predictions                

if __name__ == "__main__":

    predictions = _predict()
    plot_predictions(predictions)
    
    # pdf saved to the results folder root