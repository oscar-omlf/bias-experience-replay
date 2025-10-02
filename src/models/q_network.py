from typing import Sequence
import torch
import torch.nn as nn


_ACT = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_sizes=(128, 128), activation="relu"):
        super().__init__()
        act_cls = _ACT.get(activation, nn.ReLU)

        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), act_cls()]
            last = h

        layers += [nn.Linear(last, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
