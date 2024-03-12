import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.nn import MSELoss 
from torch.optim import Adam
from .tab import Get_data  


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


class Experiment(pl.LightningModule):
    """LightningModule to perform the actual training and data initialization"""
    def __init__(self, model, train_path, groupby_col, target_col, check_path, lr, sequence_length, weight_decay, batch_size, prediction_window, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['check_path', 'model'])
        self.model = model
        self.lr = lr
        self.sequence_length = sequence_length
        self.train_path = train_path
        self.check_path = check_path
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.target_col = target_col
        self.groupby_col = groupby_col
        self.prediction_window = prediction_window


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss)
        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log('test_loss', loss)
        return loss


    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)
        return y_hat


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
        # x, y = batch
        # preds = self.model(x)
        # return preds
        # Monte Carlo dropout inference example:
        # https://lightning.ai/docs/pytorch/stable/deploy/production_basic.html


    def predict_dataloader(self):
        pass


    def train_dataloader(self):
        pass


    def val_dataloader(self):
        pass


    def test_dataloader(self):
        pass


    def loss_function(self, inp, tgt):
        loss_fn = MSELoss()
        return loss_fn(inp, tgt)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def setJsonData(self, data: pd.DataFrame):
        self.dm = Get_data(
            self.train_path,
            self.groupby_col,
            self.target_col,
            self.prediction_window,
            self.sequence_length,
            self.batch_size
        ).setJsonData(data)

        return self
