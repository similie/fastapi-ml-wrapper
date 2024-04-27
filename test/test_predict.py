from src.prediction_models.rainfall_predictor import predict
from src.prediction_models.rainfall_predictor.utils import (reload_model,
    standard_transform,
    rescale_predictions,
    concatenate_latent_representation,
    gen_pred_dataset)
from src.prediction_models.rainfall_predictor.dataset import load_dataframe


def test_predict(data: list[str]):
    """
    Provide a json object of test data to
    test the predict pathway.
    """
    encoder = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')
    
    pr_data = load_dataframe(data)
    X, y = gen_pred_dataset(pr_data)

    X_s = standard_transform(X)
    X_s_ = concatenate_latent_representation(encoder, X_s)

    predictions = fc_model.predict(X_s_)

    assert len(predictions) == len(pr_data)


