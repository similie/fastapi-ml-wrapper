from ..src.prediction_models.rainfall_predictor.utils import reload_model


def test_load_models():
    model = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')

    assert model is not None
    assert fc_model is not None


