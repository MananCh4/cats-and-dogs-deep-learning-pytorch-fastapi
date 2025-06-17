# model_def.py

import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, n_features=784, n_classes=2):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 128),  # Input to hidden layer
            nn.ReLU(),                   # Activation
            nn.Linear(128, 64),          # Hidden to hidden
            nn.ReLU(),                   # Activation
            nn.Linear(64, n_classes)     # Hidden to output
        )

    def forward(self, x):
        return self.model(x)
