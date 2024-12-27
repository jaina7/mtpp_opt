import os, sys
sys.path.append('..')
sys.path.append('../moflow')

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, depth, output_dim, p=0.1):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        hidden_dim = 4096
        for i in range(depth-1):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
            if i % 2 == 0 and i > 0:
                hidden_dim = hidden_dim // 2
        hidden_dim = 256
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.dropout = nn.Dropout(p)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.dropout(x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x
    
class ResidualBlock(nn.Module):

    def __init__(self, in_features, p=0.1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class ResNet(nn.Module):

    def __init__(self, input_dim, depth, output_dim, p=0.1):
        super(ResNet, self).__init__()

        layers = []
        hidden_dim = 4096
        for i in range(depth-1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(ResidualBlock(hidden_dim, p))
            input_dim = hidden_dim
            if i % 2 == 0 and i > 0:
                hidden_dim = hidden_dim // 2
        hidden_dim = 256
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class WRN(nn.Module):

    def __init__(self, input_dim, depth, output_dim, p=0.1):
        super(WRN, self).__init__()

        layers = []
        for i in range(depth-1):
            layers.append(ResidualBlock(input_dim, p))
        layers.append(nn.Linear(input_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = ResNet(128, 8, 1)
    print(model)
