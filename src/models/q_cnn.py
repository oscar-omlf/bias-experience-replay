import torch
import torch.nn as nn

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

        conv_channels = tuple(conv_channels)
        if len(conv_channels) < 1:
            raise ValueError("conv_channels must have at least one element, e.g. [16] or [16, 32].")

        layers = []
        prev_ch = in_channels
        for out_ch in conv_channels:
            layers += [nn.Conv2d(prev_ch, out_ch, kernel_size=3, stride=1, padding=1), act()]
            prev_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # Spatial stays 10x10 (as you assumed before)
        last_ch = conv_channels[-1]
        flat_dim = last_ch * 10 * 10

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden), act(),
            nn.Linear(fc_hidden, n_actions),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        q = self.head(z)
        return q
