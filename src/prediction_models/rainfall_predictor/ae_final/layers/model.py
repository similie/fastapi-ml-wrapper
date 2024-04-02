import os
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
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.empty(2, x.size(0), self.latent_dim)
        nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))
        # Initialize cell state
        c0 = torch.empty(2, x.size(0), self.latent_dim)
        nn.init.xavier_uniform_(c0, gain=nn.init.calculate_gain('relu'))

        x, (h1, c1) = self.lstm(x, (h0, c0))
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
        self.lstm = nn.LSTM(input_size=latent_dim, 
            hidden_size=latent_dim,
            num_layers=2,
            dropout=dropout,
            bidirectional=False,
            batch_first=True)
        act_fn()
        self.linear = nn.Linear(latent_dim, output_size) # 128//18 = 7

    def forward(self, x):
        h0 = torch.empty(2, x.size(0), self.latent_dim)
        nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))
        # Initialize cell state
        c0 = torch.empty(2, x.size(0), self.latent_dim)
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

class Forecaster(pl.LightningModule):
    def __init__(self,
                 input_size : int,
                 latent_dim : int,
                 dropout : float,
                 ae_checkpoint_path : str,
                 act_fn : object = nn.GELU,
                 autoencoder_class : object = Autoencoder):
        """
            Inputs:
                - input_size : numbers of features
                            + latent representation
                - latent_dim : hidden layer size
                - dropout : layer normalization
                - act_fn : activation function 
        """
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.latent_dim = latent_dim
        self.save_hyperparameters()
        self.autoencoder = autoencoder_class.load_from_checkpoint(ae_checkpoint_path)
        self.autoencoder.freeze()
        self.lstm = nn.LSTM(input_size=latent_dim+input_size, 
            hidden_size=latent_dim, 
            num_layers=2,
            dropout=dropout,
            bidirectional=False, 
            batch_first=True)
        act_fn()
        self.linear = nn.Linear(self.latent_dim, self.latent_dim//2)
        act_fn()
        self.linear1 = nn.Linear(self.latent_dim//2, self.latent_dim//4)
        act_fn()
        self.linear_out = nn.Linear(self.latent_dim//4, 1)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        if self.training == False:
            h0 = torch.empty(x.size(0), self.latent_dim)
            nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))
            # Initialize cell state
            c0 = torch.empty(x.size(0), self.latent_dim)
            nn.init.xavier_uniform_(c0, gain=nn.init.calculate_gain('relu'))
        else:
            h0 = torch.empty(2, x.size(0), self.latent_dim)
            nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))
            # Initialize cell state
            c0 = torch.empty(2, x.size(0), self.latent_dim)
            nn.init.xavier_uniform_(c0, gain=nn.init.calculate_gain('relu'))
        y, (_, _) = self.autoencoder.encoder.lstm(x)
        x = torch.cat((x, y), dim=-1)
        x, (h1, c1) = self.lstm(x, (h0, c0))
        x = self.linear(x)
        x = self.linear1(x)
        return self.linear_out(x)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
    
    def training_step(self, batch):
        inputs, target = batch
        loss = self.loss_function(inputs, target)
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch
        loss = self.loss_function(inputs, target)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch
        loss = self.loss_function(inputs, target)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch
        preds = self(inputs) # target
        new_features = self.autoencoder(torch.cat((target, inputs), dim=-1))
        return preds, new_features
        
        #return self(batch[0].unsqueeze(0))        

    def loss_function(self, inputs, target):
        loss = F.mse_loss(inputs, target)
        return loss
    
    def _reconstruction_loss(self, batch):
        """
        Per batch of data, this yields reconstruction loss : MSE
        """
        x, x_f = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_f, x_hat, reduction='none')
        loss = loss.sum(dim=[2]).mean(dim=[1])
        return loss
