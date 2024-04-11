from os import path, getcwd
# model imports
from layers.model import Autoencoder, Forecaster 
from dataset import data_module
from preprocessor import load_data_csv
from mutils import get_checkpoint_filepath
from from_pretrained import autoencoder_from_pretrained

from AllWeatherConfig import AllWeatherMLConfig

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger

def _train(config: AllWeatherMLConfig) -> tuple[Autoencoder | Forecaster, dict[str, str]]:
    """
    Train the Autoencoder. Checkpoints are saved in the results
    folder. Args: - prefix_str: FC or AE
                  - latent_dim: latent dimension
                  - epochs: max epochs ( due to the seperation 
                  of station data, we cannot use early stopping )
    """
    # from the dotENV file
    lstm_config = config.lstm_config
    experiment_config = config.experiment_config
    trainer_config = config.trainer_config
    prefix_str = lstm_config.prefix
    latent_dim = lstm_config.latent_dim
    input_size = lstm_config.input_size
    output_size = lstm_config.output_size
    dropout = lstm_config.dropout
    accelerator = trainer_config.accelerator
    root_dir = trainer_config.default_root_dir

    data_path = experiment_config.data_path
    prefixes = ["FC", "AE"]
    if lstm_config.prefix not in prefixes:
        raise ValueError("Invalid prefix. Expected one of: %s" % prefixes)
    df = load_data_csv(data_path)
    CHECKPOINTPATH = path.join(getcwd(), root_dir)
    csv_logger = CSVLogger(CHECKPOINTPATH, name=f"{prefix_str}_model{latent_dim}")
    if prefix_str == "AE":
        dm = data_module(data=df)
    elif prefix_str == "FC":
        dm = data_module(data=df,
                        features=lstm_config.features, 
                        target=lstm_config.target)
        ae_checkpoint_path = get_checkpoint_filepath(model_prefix="AE", 
            latent_dim=latent_dim, 
            checkpoint_path=config.pretrain_path)
    else:
        print("Please provide FC or AE as a prefix string.")
        
    dm.setup(stage='fit')
    
    train_loader = dm.train_dataloader
    val_loader = dm.val_dataloader
    test_loader = dm.test_dataloader
    
    # Create a PyTorch Lightning trainer with the checkpoint callback
    trainer = pl.Trainer(default_root_dir=path.join(getcwd(), 
        root_dir),
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=True,
            logger=csv_logger,
            max_epochs=config.trainer_config.epochs,
            callbacks=[ModelCheckpoint(save_weights_only=True)])
    if prefix_str == "AE":
        if experiment_config.retrain_flag:
            cp = get_checkpoint_filepath(model_prefix=prefix_str,
                latent_dim=latent_dim,
                pretrain_path=config.pretrain_path)
            model = autoencoder_from_pretrained(cp)
            model.unfreeze()
        else:
            model = Autoencoder(input_size=input_size, 
                latent_dim=latent_dim,
                dropout=0.7,
                output_size=input_size,
                batch_size=1)
    elif prefix_str == "FC":
        model = Forecaster(input_size=input_size, 
            latent_dim=latent_dim,
            dropout=dropout,
            output_size=output_size,
            ae_checkpoint_path=ae_checkpoint_path)

    trainer.fit(model, train_loader, val_loader)
    
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
