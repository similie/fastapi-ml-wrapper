from src.prediction_models.rainfall_predictor.utils import reload_model


def test_load_models(path: str):
    model = reload_model(path)
    # model.summary()
    assert model is not None

if __name__ == "__main__":

    test_load_models('encoder.keras')
    test_load_models('forecaster.keras')
