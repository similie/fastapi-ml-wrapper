from src.prediction_models.rainfall_predictor.utils import reload_model


def test_load_models():
    prefix = 'src/prediction_models/rainfall_predictor'
    model = reload_model('encoder.keras', prefixFolder=prefix)
    # fc_model = reload_model('forecaster.keras', prefixFolder=prefix)

    assert model is not None
    # assert fc_model is not None


def test_shapes():
    prefix = 'src/prediction_models/rainfall_predictor'
    # Load model checkpoints
    encoder = reload_model('encoder.keras', prefixFolder=prefix)
    # fc_model = reload_model('forecaster.keras', prefixFolder=prefix)

    assert encoder.layers[0].batch_shape[1:] == (12, 9)
    # assert fc_model.layers[0].batch_shape[1:] == (12, 137)
