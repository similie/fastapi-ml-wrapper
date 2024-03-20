import os
import json
import math
import numpy as np

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
    
class Encoder(nn.Module):
    def __init__(self,
                 input_size : int,
                 latent_dim : int,
                 dropout: float,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - input_size : number of features
            - latent_dim : Dimensionality of latent representation z
            - dropout : layer normalization
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(input_size=input_size, 
            hidden_size=latent_dim, 
            num_layers=2,
            dropout=dropout,
            bidirectional=False, 
            batch_first=True)
        act_fn()
        self.lstm1 = nn.LSTM(input_size=latent_dim,
            hidden_size=latent_dim//4,
            num_layers=2,
            dropout=dropout,
            bidirectional=False,
            batch_first=True)
        act_fn()
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.empty(2, x.size(0), self.latent_dim)
        nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))
        # Initialize cell state
        c0 = torch.empty(2, x.size(0), self.latent_dim)
        nn.init.xavier_uniform_(c0, gain=nn.init.calculate_gain('relu'))
        h1 = torch.empty(2, x.size(0), self.latent_dim//4)
        nn.init.xavier_uniform_(h1, gain=nn.init.calculate_gain('relu'))
        # Initialize cell state
        c1 = torch.empty(2, x.size(0), self.latent_dim//4)
        nn.init.xavier_uniform_(c1, gain=nn.init.calculate_gain('relu'))
        
        x, (_, _) = self.lstm(x)
        x, (_, _) = self.lstm1(x)
        return x

class Decoder(nn.Module):

    def __init__(self,
                 latent_dim : int,
                 output_size: int,
                 dropout: float,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - latent_dim : Dimensionality of latent representation z
            - dropout : layer normalization
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(input_size=latent_dim//4, 
            hidden_size=latent_dim,
            num_layers=3,
            dropout=dropout,
            bidirectional=False,
            batch_first=True)
        act_fn()
        self.linear = nn.Linear(latent_dim, output_size) # 128//18 = 7

    def forward(self, x):
        h0 = torch.empty(3, x.size(0), self.latent_dim)
        nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))
        # Initialize cell state
        c0 = torch.empty(3, x.size(0), self.latent_dim)
        nn.init.xavier_uniform_(c0, gain=nn.init.calculate_gain('relu'))
        x, (c1, h1) = self.lstm(x, (c0, h0))
        x = self.linear(x)
        return x
        
class Autoencoder(pl.LightningModule):

    def __init__(self,
                 input_size: int,
                 latent_dim: int,
                 dropout: float,
                 output_size: int,
                 batch_size: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(input_size, latent_dim, dropout)
        self.decoder = decoder_class(latent_dim, output_size, dropout)
        # Example input array needed for visualizing the graph of the network
        self.test_input = [torch.randn(batch_size, 12, output_size), torch.randn(batch_size, 12, output_size)] 
               
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _reconstruction_loss(self, batch):
        """
        Per batch of data, this yields reconstruction loss : MSE
        """
        x, x_f = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_f, x_hat, reduction='none')
        loss = loss.sum(dim=[2]).mean(dim=[1])
        # loss = loss.mean(dim=[1]).sum(dim=[1]) # this needs TESTING
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(self.parameters(), lr=1e-3)
    #     # Using a scheduler is optional but can be helpful.
    #     # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                      mode='min',
    #                                                      factor=0.2,
    #                                                      patience=20,
    #                                                      min_lr=5e-5)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": f"val_loss/dataloader_idx_{dataloader_idx}"}

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._reconstruction_loss(batch)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0].unsqueeze(0))        
