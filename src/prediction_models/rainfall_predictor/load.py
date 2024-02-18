# Standard libraries
from typing import Sequence, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np

# pytorch
import torch
import torch.utils.data as data

# Utilities for working with Pytorch dataloader


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def create_data_loaders(
    *datasets: Sequence[data.Dataset],
    train: Union[bool, Sequence[bool]] = True,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Creates data loaders.

    Args:
      datasets: Datasets for which data loaders are created.
      train: Sequence indicating which datasets are used for
        training and which not. If single bool, the same value
        is used for all datasets.
      batch_size: Batch size to use in the data loaders.
      num_workers: Number of workers for each dataset.
      seed: Seed to initialize the workers and shuffling with.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            collate_fn=numpy_collate,
            num_workers=num_workers,
            persistent_workers=is_train,
            generator=torch.Generator().manual_seed(seed),
        )
        loaders.append(loader)
    return loaders


def create_dataset(df):
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features
