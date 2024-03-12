from os import path
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    AutoEncoder for TabulaRasa
    """
    def __init__(self,
                 name, 
                 n_features, 
                 hidden_size, 
                 sequence_length, 
                 batch_size,
                 num_layers,
                 dropout, 
                 bidirectional=False,
                 check_path=None):
        super(AutoEncoder, self).__init__()
        self.name = name
        self.check_path = check_path
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout,
                            bidirectional=bidirectional, 
                            batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size//4,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size//4, 1)
        self.init_weights()

    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        x, (_, _) = self.lstm2(x)
        if not self.training:
            return x
        x = self.linear(x[:,-1].unsqueeze(0))
        return x

    def init_weights(self):
        """
        Reproduce Keras default initialization for consistency w TF
        """
        ih = (
            param.data for name, param in self.named_parameters() if "weight_ih" in name
        )
        hh = (
            param.data for name, param in self.named_parameters() if "weight_hh" in name
        )
        b = (param.data for name, param in self.named_parameters() if "bias" in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
        if self.training:
            nn.init.xavier_uniform_(self.linear.weight.data)


class Forecaster(nn.Module):
    """
    Batch shape should be batch size, sequence length, 
    and latent space of the autoencoder + features, 
    ie. 128 + 10 if there are 10 features and a latent
    space of 128. 
    """
    def __init__(self,
                 name, 
                 n_features, 
                 hidden_size, 
                 sequence_length, 
                 batch_size,
                 num_layers, 
                 dropout,
                 bidirectional=False,
                 check_path=None):
        super(Forecaster, self).__init__()
        self.name = name
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.check_path = check_path
        self.lstm1 = nn.LSTM(input_size=38, # LKG
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size//4,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)
        self.linear1 = nn.Linear(hidden_size//4, 32)
        self.linear2 = nn.Linear(hidden_size//4, 1)

        self.init_weights()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=-1)
        x, (_, _) = self.lstm1(x)
        x, (_, _) = self.lstm2(x)
        x = self.linear1(x)
        return self.linear2(x)

    def init_weights(self):
        """
        Reproduce Keras default initialization for consistency
        """
        ih = (
            param.data for name, param in self.named_parameters() if "weight_ih" in name
        )
        hh = (
            param.data for name, param in self.named_parameters() if "weight_hh" in name
        )
        b = (param.data for name, param in self.named_parameters() if "bias" in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.xavier_uniform_(self.linear2.weight.data)


class TabulaRasa(nn.Module):
    """
    (Pre-Trained) AutoEncoder + Forecaster
    """
    def __init__(self, 
                 name,
                 n_features, 
                 hidden_size, 
                 sequence_length, 
                 batch_size,
                 num_layers,
                 dropout, 
                 bidirectional=False,
                 check_path=None):
        super(TabulaRasa, self).__init__()
        self.name = name
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.check_path = check_path
        self.encoder = AutoEncoder(
            name,
            n_features,
            hidden_size,
            sequence_length,
            batch_size,
            num_layers,
            dropout,
            bidirectional=False,
            check_path=None)
        self.check_path = check_path
        if path.isfile(self.check_path):
            checkpoint = torch.load(self.check_path)
            self.encoder.load_state_dict(checkpoint['state_dict'])
            self.encoder.training = False
            self.encoder.eval()
        self.forecaster = Forecaster(
            name,
            n_features,
            hidden_size, 
            sequence_length, 
            batch_size,
            num_layers,
            dropout)
        
    def forward(self, x):
        y = self.encoder(x)
        x = self.forecaster(x, y)
        return x


class ptAutoEncoder(nn.Module):
    """
    Pretrained AutoEncoder for TabulaRasa
    """
    def __init__(self,
                 name,
                 check_path):
        self.name = name
        self.check_path = check_path
        self.checkpoint = torch.load(self.check_path)
        self.encoder.training = False
        self.init_weights()
        self.encoder.freeze()

    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        x, (_, _) = self.lstm2(x)
        return x

    def init_weights(self):
        """
        Reproduce Keras default initialization for consistency w TF
        """
        ih = (
            param.data for name, param in self.named_parameters() if "weight_ih" in name
        )
        hh = (
            param.data for name, param in self.named_parameters() if "weight_hh" in name
        )
        b = (param.data for name, param in self.named_parameters() if "bias" in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
        if self.training:
            nn.init.xavier_uniform_(self.linear.weight.data)


class ptForecaster():
    """
    Pretrained Forecaster
    """
    def __init__(self,
                 name):
        super(ptForecaster, self).__init__()
        self.forecaster
        self.name = name
        self.forecaster.freeze()
        self.forecaster.training = False
        
        self.init_weights()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=-1)
        x, (_, _) = self.lstm1(x)
        x, (_, _) = self.lstm2(x)
        x = self.linear1(x)
        return self.linear2(x)

    def init_weights(self):
        """
        Reproduce Keras default initialization for consistency
        """
        ih = (
            param.data for name, param in self.named_parameters() if "weight_ih" in name
        )
        hh = (
            param.data for name, param in self.named_parameters() if "weight_hh" in name
        )
        b = (param.data for name, param in self.named_parameters() if "bias" in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.xavier_uniform_(self.linear2.weight.data)


class ptTabulaRasa(nn.Module):
    """
    Everything pretrained
    """
    def __init__(self, 
                 name):
        super(TabulaRasa, self).__init__()
        self.name = name
        self.encoder
        self.forecaster
        
    def forward(self, x):
        y = self.encoder(x)
        x = self.forecaster(x, y)
        return x
