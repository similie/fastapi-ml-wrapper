import json
from src.prediction_models.rainfall_predictor.dataset import (gen_pred_dataset,
    standard_transform,
    standard_inverse_transform,
    onehot_transform,
    max_transform,
    max_inverse_transform)
from src.prediction_models.rainfall_predictor.preprocessor import load_dataframe
from src.prediction_models.rainfall_predictor.AllWeatherConfig import getAllWeatherMLConfig

if __name__ == "__main__":

    # get prediction window to test shapes
    # of model inputs and datasets
    config = getAllWeatherMLConfig()
    prediction_window = config.experiment_config.prediction_window
 
    # test data
    with open('./tmp/all_weather_cube_query_response.json') as f:
        d = json.load(f)
        weather_data = d['data']
    # load test data
    data = load_dataframe(weather_data)
    X_p, y_p = gen_pred_dataset(data, prediction_window)
    # test shape of concatenated latent space `X_s_`
    X_x = standard_transform(X_p)
    X_o = onehot_transform(X_p)
    X_inv = standard_inverse_transform(X_x)
    y_s = max_transform(y_p)
    y_inv = max_inverse_transform(y_s)
    
    assert X_inv == X_p

    assert y_p == y_inv
    
