from os import path, getcwd
from preprocessor import load_data_json
from predict import _predict

json_path = path.join(getcwd(), '../project/checkpoints/test_cube.json') 

def check_prediction_inputs(json_path):
    data = load_data_json(json_path)
    return data
    
data = check_prediction_inputs(json_path)
predictions = _predict(data)
