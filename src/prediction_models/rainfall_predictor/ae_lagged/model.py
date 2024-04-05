from layers import Recurrent
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
  print("Install lightning...")

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class Lagged(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 latent_dim: int,
                 dropout: float,
                 output_size: int,
                 batch_size: int,
                 lstm_class : object = Recurrent):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.lstm = lstm_class(input_size, latent_dim, output_size, dropout)
        self.test_input = [torch.randn(batch_size, 12, input_size), torch.randn(batch_size, 12, 1)] 
               
    def forward(self, x):
        return self.lstm(x)
    # def configure_optimizers(self):
    #     return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch):
        inputs, target = batch
        loss = self.loss_function(inputs, target)
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        loss = self.loss_function(inputs, target)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        loss = self.loss_function(inputs, target)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx):
        inputs, target = batch
        return self(inputs)

    def loss_function(self, inputs, target):
        loss = F.mse_loss(inputs, target)
        return loss
