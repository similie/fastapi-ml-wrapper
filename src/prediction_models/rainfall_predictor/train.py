from os import path, getcwd
# model imports
from layers.model import Autoencoder, Forecaster 
from dataset import data_module
from preprocessor import load_data_csv
from mutils import (get_checkpoint_filepath, 
                    autoencoder_from_pretrained,
                    forecaster_from_pretrained)

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger

def _train(prefix_str: str,
    latent_dim: int,
    epochs: int) -> tuple[Autoencoder | Forecaster, dict[str, any]]:
    """
    Train the Autoencoder. Checkpoints are saved in the results
    folder. Args: - prefix_str: FC or AE
                  - latent_dim: latent dimension
                  - epochs: max epochs ( due to the seperation 
                  of station data, we cannot use early stopping )
    Set retrain to `True` (below) if you want to retrain a model
    Returns: trained model and training metrics in a dict
    """
    retrain = False  
    trainer_dir = 'results'
    data_path = '/home/leigh/Code/ekoh/tabula_rasa/data/combined.csv'
    prefixes = ["FC", "AE"]
    if prefix_str not in prefixes:
        raise ValueError("Invalid prefix. Expected one of: %s" % prefixes)
    df = load_data_csv(data_path)
    CHECKPOINTPATH = path.join(getcwd(), trainer_dir)
    csv_logger = CSVLogger(CHECKPOINTPATH, name=f"{prefix_str}_model{latent_dim}")
    if prefix_str == "AE":
        dm = data_module(data=df)
    elif prefix_str == "FC":
        dm = data_module(data=df, 
                        target=['precipitation'])
        ae_checkpoint_path = get_checkpoint_filepath(model_prefix="AE", 
            latent_dim=latent_dim, 
            checkpoint_path=CHECKPOINTPATH)
        
    dm.setup(stage='fit')
    
    train_loader = dm.train_dataloader
    val_loader = dm.val_dataloader
    test_loader = dm.test_dataloader
    
    # Create a PyTorch Lightning trainer with the checkpoint callback
    trainer = pl.Trainer(default_root_dir=CHECKPOINTPATH,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=True,
        logger=csv_logger,
        max_epochs=epochs,
        callbacks=[ModelCheckpoint(save_weights_only=True)])
    if prefix_str == "AE":
        if retrain:
            cp = get_checkpoint_filepath(model_prefix=prefix_str,
                latent_dim=latent_dim,
                checkpoint_path=CHECKPOINTPATH)
            model = autoencoder_from_pretrained(cp)
            model.unfreeze()
        else:
            model = Autoencoder(input_size=7, 
                latent_dim=latent_dim,
                dropout=0.7,
                output_size=7,
                batch_size=1)
    elif prefix_str == "FC":
        model = Forecaster(input_size=7, 
            latent_dim=latent_dim,
            dropout=0.5,
            output_size=1,
            ae_checkpoint_path=ae_checkpoint_path)
    trainer.fit(model, train_loader, val_loader)
    
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
