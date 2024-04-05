from os import path, listdir, getcwd
# model imports
from model import Lagged
from dataset import LaggedData 
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
  print("Install lightning...")
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger

CHECKPOINTPATH = path.join(getcwd(), './results')


def _train(latent_dim: int, station_id: str, dm: pl.DataModule):

    """
        Train the LSTM.   Checks for a pre-existing model
        checkpoint for the given latent dimension and loads 
        that if it exists. Checkpoints are saved in the results
        folder. Args: latent_dimension, station_id to pass to data
        module.
    """
    # Set up data module for the station
    dm.setup(station_id=station_id, stage=None)

    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    val_loader = dm.test_dataloader()

    csv_logger = CSVLogger(CHECKPOINTPATH, name=f"Lag_model{latent_dim}")
    MODEL_PATH = path.join(getcwd(), f"results/Lag_model{latent_dim}/version_0/checkpoints/")
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=path.join(CHECKPOINTPATH, f"Lag_model{latent_dim}"),
                         accelerator="cpu",
                         devices=1,
                         enable_checkpointing=True,
                         logger=csv_logger,
                         max_epochs=3,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    LearningRateMonitor("epoch")])
    # Check whether pretrained model exists. If yes, load it and skip training
    try:
        checkpoint_path = path.join(MODEL_PATH, listdir(MODEL_PATH)[0])
        if path.isfile(checkpoint_path):
            print("Found pretrained Autoencoder.")
            model = Lagged.load_from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print("Loading model...")    
        model = Lagged(input_size=48, 
                        latent_dim=latent_dim,
                        dropout=0.7,
                        output_size=1,
                        batch_size=12,
                        )
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
