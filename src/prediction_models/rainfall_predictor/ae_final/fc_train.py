from os import path, getcwd, listdir

from layers.model import Autoencoder, Forecaster 
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
  print("Install lightning...")
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger

CHECKPOINTPATH = path.join(getcwd(), "results")
def _train(latent_dim, dm):
    """
        Train the forecaster. Checks for a pre-existing model
        checkpoint for the given latent dimension, and loads 
        that instead if it exists.
    """
    csv_logger = CSVLogger(CHECKPOINTPATH, name=f"FC_model{latent_dim}")
    train_loader = dm.train_combined_loader()
    val_loader = dm.val_combined_loader()
    test_loader = dm.test_combined_loader()
    # Create a PyTorch Lightning trainer with the generation callback
    AE_MODEL_PATH = path.join(getcwd(), f"results/AE_model{latent_dim}/version_0/checkpoints/")
    FC_MODEL_PATH = path.join(getcwd(), f"results/FC_model{latent_dim}/version_0/checkpoints/")
    try:
        ae_checkpoint_path = path.join(AE_MODEL_PATH, listdir(AE_MODEL_PATH)[0])
        if path.isfile(ae_checkpoint_path):
            print("Found pretrained Autoencoder.")
    except FileNotFoundError:
        print("No pretrained Forecaster found...")    
    trainer = pl.Trainer(default_root_dir=CHECKPOINTPATH,
                         accelerator="cpu",
                         devices=1,
                         # enable_checkpointing=False,
                         logger=csv_logger,
                         max_epochs=3,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    LearningRateMonitor("epoch")])
    # Check whether pretrained model exists. If yes, load it and skip training
    try:
        pretrained_filename = path.join(FC_MODEL_PATH, listdir(FC_MODEL_PATH)[0])
        if path.isfile(pretrained_filename):
            print("Found pretrained Forecaster, loading...")
            model = Forecaster.load_from_checkpoint(pretrained_filename)
    except FileNotFoundError:
        print("No pretrained Forecaster found...")
        model = Forecaster(input_size=7, 
                            latent_dim=latent_dim,
                            dropout=0.5,
                            ae_checkpoint_path=ae_checkpoint_path)

        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

 
