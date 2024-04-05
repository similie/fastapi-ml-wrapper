import torch
import torch.nn as nn

class Recurrent(nn.Module):
    def __init__(self,
                 input_size : int,
                 latent_dim : int,
                 dropout: float,
                 output_size: int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - input_size : number of features
            - latent_dim : Dimensionality of latent representation z
            - dropout : layer normalization
            - output_size: output dimension
            - act_fn : Activation function used 
                throughout the encoder network
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=input_size, 
            hidden_size=latent_dim, 
            num_layers=2,
            dropout=dropout,
            bidirectional=False, 
            batch_first=True)
        act_fn()
        self.linear = nn.Linear(self.latent_dim, self.output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.empty(2, 12, self.latent_dim)
        nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))
        # Initialize cell state
        c0 = torch.empty(2, 12, self.latent_dim)
        nn.init.xavier_uniform_(c0, gain=nn.init.calculate_gain('relu'))

        x, (h1, c1) = self.lstm(x, (h0, c0))
        x = self.linear(x)
        return x

