from src.prediction_models.rainfall_predictor.predict import predict
from test.fixtures.all_weather import loadJsonFixture, serialiseToML


def test_predict():
    """
    Provide a json object of sample test data to
    test the predict pathway.
    """
    jsonData = loadJsonFixture()
    data = serialiseToML(jsonData)
    predictions = predict(weather_data=data)
    assert predictions is not None

    # weird side effect of missing 22% coverage for the debug flag.
    predictions = predict(weather_data=data, debug=True)
    assert predictions is not None
