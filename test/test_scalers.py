import pytest
# from src.interfaces.ReqRes import BasePostRequest
# from src.prediction_models import RainfallPredictor
# from src.prediction_models.rainfall_predictor.PredictionPostRequests import (
#     ForecastPredictionPostRequest,
#     CubePredictionPostRequest
# )

# from ..src.prediction_models.\
#     rainfall_predictor.AllWeatherConfig import (getAllWeatherMLConfig,
#     getAllWeatherConfig)

# from ..src.prediction_models.rainfall_predictor.\
#     PredictionPostRequests import ForecastPredictionPostRequest
# from ..src.prediction_models.rainfall_predictor.\
#     AllWeatherCubeResponse import (cleanCubeNameFromResponseKeys,
#     AllWeatherCubeQueryResponse)
# from ..src.prediction_models.rainfall_predictor.dataset import (standard_transform,
#     max_transform, onehot_transform, gen_pred_dataset, max_inverse_transform,
#     standard_inverse_transform)
# from ..src.prediction_models.rainfall_predictor.preprocessor import load_dataframe


@pytest.mark.skip(reason='TODO')
def test_scalers():
    pass

# def loadJsonFixture():
#     '''
#     load the sample Json file to the Cube query resonse model format.
#     '''
#     p = path.join(getcwd(),
#         'test',
#         'fixtures',
#         'all_weather_cube_query_response.json')
#     with open(p, 'r') as file:
#         jsonData = json.load(file)
#         return json.dumps(jsonData)

# def serialise_to_ml():
#     jsonData = loadJsonFixture()
#     # cubeName = getAllWeatherConfig().cube_name
#     cleanedJson = cleanCubeNameFromResponseKeys(jsonData)
#     jsonData = json.loads(cleanedJson)
#     model = AllWeatherCubeQueryResponse.model_validate(jsonData)
#     return model.model_dump(by_alias=True)['data']

# def test_scalers():
#     """
#     Provide a json object of sample test data to
#     test the scaler functions.
#     """
#     data = serialise_to_ml()
#     weather_data = load_dataframe(data)
#     X, y = gen_pred_dataset(weather_data, 12)
#     X_s = standard_transform(X)
#     X_o = onehot_transform(X)
#     y_s = max_transform(y)
#     X_inv = standard_inverse_transform(X_s)
#     y_inv = max_inverse_transform(y_s)

#     assert X_s is not None
#     assert X_o is not None
#     assert y_s is not None

#     assert X_inv == X
#     assert y_inv == y
