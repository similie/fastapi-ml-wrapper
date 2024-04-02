from os import path, getcwd
from preprocessor import load_data_json
from predict import _predict
from mutils import plot_predictions
from dataset import get_dm

json_path = '/home/leigh/Code/ekoh/tabula_rasa/data/combined.csv' 

def check_prediction_inputs(json_path):
    data = load_data_json(json_path)
    return data

predictions = _predict(json_path)
print(predictions)
print(type(predictions))
