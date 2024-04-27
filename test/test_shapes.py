from ..src.prediction_models.rainfall_predictor.utils import reload_model


def test_shapes():    
    # Load model checkpoints
    encoder = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')

    assert encoder.layers[0].batch_shape[1:] == (12, 9)
    assert fc_model.layers[0].batch_shape[1:] == (12, 137)