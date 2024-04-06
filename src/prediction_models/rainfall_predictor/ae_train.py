from os import path, listdir, getcwd
# model imports
from .layers.model import Autoencoder 
from .dataset import data_module
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger

def _train(latent_dim: int, 
    epochs: int = 3) -> tuple[Autoencoder, dict[str, str]]:

    """
    Train the Autoencoder. Checks for a pre-existing model
    checkpoint for the given latent dimension and loads 
    that if it exists. Checkpoints are saved in the results
    folder. Args: latent_dimension, dm: the datamodule, ie
    dm = data_module()
    """
    
    CHECKPOINTPATH = path.join(getcwd(), './results')
    csv_logger = CSVLogger(CHECKPOINTPATH, name=f"AE_model{latent_dim}")
    
    dm = datamodule()
    dm.setup(stage='fit')
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=path.join(CHECKPOINTPATH, 
                        f"AE_{latent_dim}"),
                        accelerator="cpu",
                        devices=1,
                        enable_checkpointing=True,
                        logger=csv_logger,
                        max_epochs=epochs,
                        callbacks=[ModelCheckpoint(save_weights_only=True)])
    
    model = Autoencoder(input_size=7, 
                        latent_dim=latent_dim,
                        dropout=0.7,
                        output_size=7,
                        batch_size=1,)
    
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
