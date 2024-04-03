from os import path, listdir, getcwd
# model imports
from layers.model import Autoencoder 
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
  print("Install lightning...")
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger

CHECKPOINTPATH = path.join(getcwd(), './results')


def _train(latent_dim, dm):

    """
        Train the Autoencoder. Checks for a pre-existing model
        checkpoint for the given latent dimension and loads 
        that if it exists. Checkpoints are saved in the results
        folder. Args: latent_dimension, dm: the datamodule, ie
        dm = get_dm()
    """
    
    csv_logger = CSVLogger(CHECKPOINTPATH, name=f"AE_model{latent_dim}")
    AE_MODEL_PATH = path.join(getcwd(), f"results/AE_model{latent_dim}/version_0/checkpoints/")
    train_loader = dm.train_combined_loader()
    val_loader = dm.val_combined_loader()
    test_loader = dm.test_combined_loader()
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=path.join(CHECKPOINTPATH, f"AE_{latent_dim}"),
                         accelerator="cpu",
                         devices=1,
                         enable_checkpointing=True,
                         logger=csv_logger,
                         max_epochs=3,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    LearningRateMonitor("epoch")])
    # Check whether pretrained model exists. If yes, load it and skip training
    try:
        ae_checkpoint_path = path.join(AE_MODEL_PATH, listdir(AE_MODEL_PATH)[0])
        if path.isfile(ae_checkpoint_path):
            print("Found pretrained Autoencoder.")
            model = Autoencoder.load_from_checkpoint(pretrained_filename)
    except FileNotFoundError:
        print("Loading model...")    
        model = Autoencoder(input_size=7, 
                        latent_dim=latent_dim,
                        dropout=0.7,
                        output_size=7,
                        batch_size=1,
                        )
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
