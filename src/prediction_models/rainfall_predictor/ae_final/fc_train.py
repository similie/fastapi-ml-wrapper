from os import path, getcwd, listdir

from dataset import data_module
from mutils import get_checkpoint_filepath
from layers.model import Forecaster 
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger

def _train(latent_dim: int, 
            epochs: int = 3) -> tuple[Forecaster, dict[str, str] ]:
    """
    Train the forecaster. Uses a pretrained Autoencoder from
    a checkpoint. Logs metrics with CSVLogger in CHECKPOINT-
    PATH     
    """

    CHECKPOINTPATH = path.join(getcwd(), "results")
    
    ae_checkpoint_path = get_checkpoint_filepath("AE", 
                                                 latent_dim, 
                                                 CHECKPOINTPATH)
    
    dm = data_module()
    dm.setup(stage="fit")
    csv_logger = CSVLogger(CHECKPOINTPATH, name=f"FC_model{latent_dim}")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    # Create a PyTorch Lightning trainer with callback
    trainer = pl.Trainer(default_root_dir=CHECKPOINTPATH,
                         accelerator="cpu",
                         devices=1,
                         enable_checkpointing=False,
                         logger=csv_logger,
                         max_epochs=epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True)])
    model = Forecaster(input_size=7, 
                        latent_dim=latent_dim,
                        dropout=0.5,
                        ae_checkpoint_path=ae_checkpoint_path)

    trainer.fit(model, train_loader, val_loader)
    
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

