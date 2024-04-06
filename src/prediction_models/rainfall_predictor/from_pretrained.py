from layers.model import Forecaster, Autoencoder
import pytorch_lightning as pl

def forecaster_from_pretrained(checkpoint_path: str) -> pl.LightningModule:
    model = Forecaster.load_from_checkpoint(checkpoint_path)
    model.freeze()
    return model

def autoencoder_from_pretrained(checkpoint_path: str) -> pl.LightningModule:
    model = Autoencoder.load_from_checkpoint(checkpoint_path)
    model.freeze()
    return model
