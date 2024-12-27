import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class PreProcess(nn.Module):
    """
    Preprocess the data for the property prediction task. The data is divided into 5 categories:
    Identity, Positive, Negative, Exponential, and Bounded.
    The forward function takes in the data as torch tensor and returns the standardized preprocessed data.
    Mean and variance are calculated for each category and used to standardize the data.
    """

    def __init__(self, identity_idxs, positive_idxs, negative_idxs, exponential_idxs, bonuded_idxs):
        super(PreProcess, self).__init__()
        self.identity_idxs = identity_idxs
        self.positive_idxs = positive_idxs
        self.negative_idxs = negative_idxs
        self.exponential_idxs = exponential_idxs
        self.bonuded_idxs = bonuded_idxs

    def inv_sigmoid(self, x, upper_bound, lower_bound):
        output = (x - lower_bound) / (upper_bound - x)
        return torch.log(output)

    def inv_log_exp(self, x, eps=1e-6):
        return torch.log(torch.exp(x) - 1 + eps)
    
    def neg_inv_log_exp(self, x, eps=1e-6):
        return torch.log(torch.exp(-x) - 1 + eps)
    
    def sigmoid(self, x, upper_bound, lower_bound):
        return lower_bound + (upper_bound - lower_bound) * torch.sigmoid(x)
    
    def log_exp(self, x):
        return torch.log(1 + torch.exp(x))
    
    def neg_log_exp(self, x):
        return -torch.log(1 + torch.exp(x))
    
    def forward(self, x):

        outputs = torch.zeros_like(x)

        # Identity transformation
        outputs[:, self.identity_idxs] = x[:, self.identity_idxs]

        # Positive transformation
        subset = x[:, self.positive_idxs]
        subset = self.inv_log_exp(subset)
        outputs[:, self.positive_idxs] = subset

        # Negative transformation
        subset = x[:, self.negative_idxs]
        subset = self.neg_inv_log_exp(subset)
        outputs[:, self.negative_idxs] = subset

        # Exponential transformation
        subset = x[:, self.exponential_idxs]
        subset = torch.log(subset + 1e-6)
        outputs[:, self.exponential_idxs] = subset

        # Bounded transformation
        if self.bounds is None: 
            self.bounds = {}
            for idx in self.bonuded_idxs:
                subset = x[:, idx]
                lower_bound = torch.min(subset) - 1e-6
                upper_bound = torch.max(subset) + 1e-6
                self.bounds[idx] = (lower_bound, upper_bound)
    
        for idx in self.bonuded_idxs:
            subset = x[:, idx]
            lower_bound, upper_bound = self.bounds[idx]
            subset = self.inv_sigmoid(subset, upper_bound, lower_bound)
            outputs[:, idx] = subset

        if self.means is None or self.stds is None:
            self.means = torch.mean(outputs, dim=0)
            self.stds = torch.std(outputs, dim=0)

        outputs = (outputs - self.means) / self.stds
        return outputs

    def inverse(self, x):
        outputs = torch.zeros_like(x)
        x = x * self.stds + self.means

        # Identity transformation
        outputs[:, self.identity_idxs] = x[:, self.identity_idxs]

        # Positive transformation
        subset = x[:, self.positive_idxs]
        subset = self.log_exp(subset)
        outputs[:, self.positive_idxs] = subset

        # Negative transformation
        subset = x[:, self.negative_idxs]
        subset = self.neg_log_exp(subset)
        outputs[:, self.negative_idxs] = subset

        # Exponential transformation
        subset = x[:, self.exponential_idxs]
        subset = torch.exp(subset)
        outputs[:, self.exponential_idxs] = subset

        # Bounded transformation
        for idx in self.bonuded_idxs:
            subset = x[:, idx]
            lower_bound, upper_bound = self.bounds[idx]
            subset = self.sigmoid(subset, upper_bound, lower_bound)
            outputs[:, idx] = subset

        return outputs
    
if __name__ == '__main__':

    x = torch.randn(10, 5)
    identity_idx = 0
    positive_idx = 1
    negative_idx = 2
    exponential_idx = 3
    bounded_idx = 4
    x[:, positive_idx] = torch.abs(torch.randn_like(x[:, positive_idx])*10)
    x[:, negative_idx] = -torch.abs(torch.randn_like(x[:, negative_idx])*10)
    x[:, exponential_idx] = torch.exp(torch.randn_like(x[:, exponential_idx])*5)
    x[:, bounded_idx] = torch.rand_like(x[:, bounded_idx])*5 + 5

    print('Original:', x)
    preprocessor = PreProcess([identity_idx], [positive_idx], [negative_idx], [exponential_idx], [bounded_idx])
    x = preprocessor(x)
    print('Mean:', preprocessor.means)
    print('Std:', preprocessor.stds)
    print('Preprocessed:', x)

    torch.save(preprocessor, 'preprocessor.pth')
    preprocessor = torch.load('preprocessor.pth')
    x = preprocessor.inverse(x)
    print('Transformed:', x)