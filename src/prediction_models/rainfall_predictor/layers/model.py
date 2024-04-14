import numpy as np
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# PyTorch Lightning
import pytorch_lightning as pl

class Encoder(nn.Module):
    """
    Inputs:
        - input_size : number of features
        - latent_dim : Dimensionality of latent representation z
        - dropout : layer normalization
        - act_fn : Activation function used throughout the encoder network
    """
    def __init__(self,
                 input_size : int,
                 latent_dim : int,
                 dropout: float,
                 act_fn : object = nn.Tanh):
        
        super().__init__()
        self.act_fn = act_fn()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(input_size=input_size, 
            hidden_size=latent_dim, 
            num_layers=2,
            dropout=dropout,
            bidirectional=False, 
            batch_first=True)
        
 
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.empty(2, x.size(0), self.latent_dim)
        # Initialize cell state
        c0 = torch.empty(2, x.size(0), self.latent_dim)

        x, (h1, c1) = self.lstm(x, (h0, c0))
        return x

class Decoder(nn.Module):

    def __init__(self,
                 latent_dim : int,
                 output_size: int,
                 dropout: float,
                 act_fn : object = nn.Tanh):
        """
        Inputs:
            - latent_dim : Dimensionality of latent representation z
            - dropout : layer normalization
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.act_fn = act_fn()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(input_size=latent_dim, 
            hidden_size=latent_dim,
            num_layers=2,
            dropout=dropout,
            bidirectional=False,
            batch_first=True)
        self.linear = nn.Linear(latent_dim, output_size) 

    def forward(self, x):
        h0 = torch.empty(2, x.size(0), self.latent_dim)
        # Initialize cell state
        c0 = torch.empty(2, x.size(0), self.latent_dim)
        x, (c1, h1) = self.lstm(x, (c0, h0))
        x = self.linear(x)
        return x
        
class Autoencoder(pl.LightningModule):
    """
        Autoencoder: trains on shifted timeseries data
        using reconstruction loss.
        Args: batch_size = 1
    """
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
        self.test_input = [torch.randn(batch_size, 12, input_size), torch.randn(batch_size, 12, output_size)] 

        # self.init_weights()

    def init_weights(self):
        """
        Reproduce Keras default init
        """
        if self.training:
            ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
            hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
            b = (param.data for name, param in self.named_parameters() if 'bias' in name)
            for t in ih:
                nn.init.xavier_uniform_(t,
                    gain=nn.init.calculate_gain(nonlinearity='linear'))
            for t in hh:
                nn.init.orthogonal_(t)
            for t in b:
                nn.init.constant_(t, 0)
        
            nn.init.xavier_uniform_(self.decoder.linear.weight,
                nn.init.calculate_gain(nonlinearity='linear'))

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

    def smape(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = lambda y, yh: 2*(y - yh).abs() / (y.abs() + yh.abs())
        return loss(y, y_hat).sum(dim=[2]).mean(dim=[1])

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
                                                          
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
                 output_size: int,
                 ae_checkpoint_path : str,
                 act_fn : object = nn.Tanh,
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
        self.act_fn = act_fn()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.save_hyperparameters()
        self.autoencoder = autoencoder_class.load_from_checkpoint(ae_checkpoint_path)
        self.autoencoder.eval()
        self.lstm = nn.LSTM(input_size=latent_dim+input_size, 
            hidden_size=latent_dim, 
            num_layers=2,
            dropout=dropout,
            bidirectional=False, 
            batch_first=True)
        self.linear = nn.Linear(self.latent_dim, self.latent_dim//2)
        self.linear1 = nn.Linear(self.latent_dim//2, self.latent_dim//4)
        self.linear_out = nn.Linear(self.latent_dim//4, self.output_size)
    
        # self.init_weights()

    def init_weights(self):
        """
        Reproduce Keras default init
        """
        if self.training:
            ih = (param.data for name, param in self.lstm.named_parameters() if 'weight_ih' in name)
            hh = (param.data for name, param in self.lstm.named_parameters() if 'weight_hh' in name)
            b = (param.data for name, param in self.lstm.named_parameters() if 'bias' in name)
            for t in ih:
                nn.init.xavier_uniform_(t,
                    gain=nn.init.calculate_gain(nonlinearity='linear'))
            for t in hh:
                nn.init.orthogonal_(t)
            for t in b:
                nn.init.constant_(t, 0)
        
            nn.init.xavier_uniform_(self.linear.weight,
                nn.init.calculate_gain(nonlinearity='linear'))
            nn.init.xavier_uniform_(self.linear1.weight,
                nn.init.calculate_gain(nonlinearity='linear'))
            nn.init.xavier_uniform_(self.linear_out.weight,
                nn.init.calculate_gain(nonlinearity='linear'))

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.empty(2, x.size(0), self.latent_dim)
        # Initialize cell state
        c0 = torch.empty(2, x.size(0), self.latent_dim)

        y = self.autoencoder.encoder(x)
        x = torch.cat((x, y), dim=-1)
        x, (h1, c1) = self.lstm(x, (h0, c0))
        x = self.act_fn(self.linear(x))
        x = self.act_fn(self.linear1(x))
        return self.linear_out(x)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
    
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch
        yhat = self(inputs)
        loss = self.loss_function(yhat, target)
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch
        yhat = self(inputs)
        loss = self.loss_function(yhat, target)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch
        yhat = self(inputs)
        loss = self.loss_function(yhat, target)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch[0].unsqueeze(0), batch[1].unsqueeze(0)
        features = self.autoencoder(inputs)
        preds = self(inputs)
        return torch.cat((preds, features[:,:,1:]),dim=-1)
        
    def loss_function(self, y_hat, target):
        loss = F.mse_loss(y_hat, target)
        return loss
    
    def smape(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = lambda y, yh: 2*(y - yh).abs() / (y.abs() + yh.abs())
        return loss(y, y_hat).mean(dims=[0])

    def scaled_reconstruction_loss(self, batch):
        """
        Per batch of data, this yields reconstruction loss : MSE
        """
        x, x_f = batch
        x_hat = self.forward(x)
        scaler = torch.linspace(0.1, 1, x.size(1))[None,:]
        loss = F.mse_loss(x_f, x_hat, reduction='none')
        scaled_loss = loss.sum(dim=[2]) * scaler
        return scaled_loss.mean(dim=[1])