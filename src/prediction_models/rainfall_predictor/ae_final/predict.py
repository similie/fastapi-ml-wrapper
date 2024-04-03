import os
# import torch
import pandas as pd
# model imports
from layers.model import Forecaster
from dataset import get_dm
from preprocessor import load_data_json
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
  print("Install lightning...")

def _predict(
        data: str,
        latent_dim: int = 64,
        load_fn=load_data_json,
        ):
    """
        Returns a dict with station numbers as keys and dataframes 
        of hourly weather station predictions. Whatever time frame 
        you supply, the system will return 6 times the length up to 
        a limit of 12 hours, or 6 * 12 = 72 hours / 3 days.
    """
    CHECKPOINTPATH = os.path.join(os.getcwd(), "results")
    try:
        FC_MODEL_PATH = os.path.join(os.getcwd(), f"results/FC_model{latent_dim}/version_0/checkpoints/")
        fc_checkpoint_path = os.path.join(FC_MODEL_PATH, os.listdir(FC_MODEL_PATH)[0])
    except FileNotFoundError:
        print("Model weights not found...")
    dm = get_dm(data_dir=data, load_fn=load_fn) 
    
    trainer = pl.Trainer(default_root_dir=CHECKPOINTPATH,
                            enable_checkpointing=False,
                            accelerator="cpu",
                            devices=1)
    
    if os.path.isfile(fc_checkpoint_path):
        print("Found pretrained model, loading...")
        model = Forecaster.load_from_checkpoint(fc_checkpoint_path)
        model.freeze()
    else:
        print("Pretrained model not found...")
    # Feed predictions into dataframes with an extended _df.index
    predictions = generate_predictions(model, trainer, dm)
    # return predictions, new_feats, model
    return predictions
def generate_predictions(model, trainer, dm, preds: dict = None):
    """
        Generate 6 * 12 hours of predictions, 
        either raw, or to be fed as a latent- 
        space representation into a second
        specialized forecaster model.
    """
    result = {}
    loader = dm.predict_combined_loader(preds=preds)
    predictions = trainer.predict(model, loader)
    result = dm.process_preds(predictions)
    for i in range(5):
        predictions = trainer.predict(model, 
                                      dm.predict_combined_loader(preds=preds))
        preds = dm.process_preds(predictions)
        for s, _df in preds.items():
            result[s] = pd.concat([result[s], _df])
    return result
