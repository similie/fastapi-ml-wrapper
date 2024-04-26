from utils import reload_model


if __name__ == "__main__":

    encoder = reload_model('encoder.keras')
    fc_model = reload_model('forecaster.keras')

    encoder.summary()
    print("\n")
    fc_model.summary()
