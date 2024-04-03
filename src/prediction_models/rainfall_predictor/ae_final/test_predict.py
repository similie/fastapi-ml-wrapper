from os import path, getcwd
from predict import _predict
from preprocessor import load_data_json

json_path = path.join(getcwd(), '../../../../test/fixtures/all_weather_cube_query_response.json')

def verify_preds_from_json(json_path, latent_dim=128, load_fn=load_data_json):
    predictions = _predict(json_path, latent_dim, load_fn=load_fn)
    assert predictions != None
    for s, _df in predictions.items():
        print("STATION #", s, "-->\n", _df.max())

verify_preds_from_json(json_path)
