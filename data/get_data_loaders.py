## Load saved arrays from disk and return them as PyTorch DataLoader objects

import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_data_loaders(save_path='./data/', batch_size=128, shuffle_train=True, validation_ratio=0.2):

    train_latents = torch.load(os.path.join(save_path, 'train_latents.pt'))
    test_latents = torch.load(os.path.join(save_path, 'test_latents.pt'))
    train_props = torch.load(os.path.join(save_path, 'train_props.pt'))
    test_props = torch.load(os.path.join(save_path, 'test_props.pt'))

    train_val_dataset = TensorDataset(train_latents, train_props)
    test_dataset = TensorDataset(test_latents, test_props)

    num_train = len(train_val_dataset)
    num_val = int(np.floor(validation_ratio * num_train))
    num_train = num_train - num_val
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader