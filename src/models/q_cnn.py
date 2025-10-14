import torch
import torch.nn as nn
from typing import Tuple


_ACT = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class CNNQNetwork(nn.Module):
    def __init__(
            self, in_channels: int,
            n_actions: int,
            conv_channels=(16, 32),
            fc_hidden=128,
            activation="relu"
        ):
        super().__init__()
        act = _ACT.get(activation, nn.ReLU)

        c1, c2 = conv_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1), act(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1), act(),
        )
        # Spatial stays 10x10
        flat_dim = c2 * 10 * 10
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden), act(),
            nn.Linear(fc_hidden, n_actions),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        z = self.encoder(x)
        q = self.head(z)
        return q
