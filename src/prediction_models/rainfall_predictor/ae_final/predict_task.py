import os
from mutils import plot_predictions
from predict import _predict
from preprocessor import load_data_json

csv_path = os.path.join(os.getcwd(), "../tabula_rasa/data/combined.csv")
json_path = '/path/to/json-file' 

predictions = _predict(json_path, 256, load_fn=load_data_json)

# Predictions from a CSV file

# prediction = _predict(csv_path, 128)

# for k, v in prediction.items():
#     print("STATION #", k, "-->", v.max())

plot_predictions(prediction) 
