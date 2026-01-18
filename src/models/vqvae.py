from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.beta = float(beta)

        self.emb = nn.Embedding(self.codebook_size, self.embed_dim)
        nn.init.uniform_(self.emb.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_e: (B, D)

        Returns:
          z_q: (B, D) quantized vectors (straight-through)
          idx: (B,) code indices
          vq_loss: scalar
        """
        # distances: ||z||^2 + ||e||^2 - 2 z*e
        z = z_e
        z2 = (z ** 2).sum(dim=1, keepdim=True)               # (B,1)
        e2 = (self.emb.weight ** 2).sum(dim=1).unsqueeze(0)  # (1,K)
        ze = z @ self.emb.weight.t()                         # (B,K)
        dist = z2 + e2 - 2.0 * ze                            # (B,K)

        idx = torch.argmin(dist, dim=1)  # (B,)
        z_q = self.emb(idx)              # (B,D)

        # losses
        # codebook loss: ||sg[z_e] - z_q||^2
        # commitment loss: beta * ||z_e - sg[z_q]||^2
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through estimator
        z_q_st = z + (z_q - z).detach()
        return z_q_st, idx, vq_loss


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, embed_dim: int, obs_shape: Tuple[int, ...]):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.embed_dim = int(embed_dim)
        self.obs_shape = tuple(obs_shape)

        if len(self.obs_shape) == 3:
            # image-like: CHW in model
            C, H, W = self.obs_shape
            self.conv = nn.Sequential(
                nn.Conv2d(C, hidden_channels, 3, stride=2, padding=1),  # H->ceil(H/2)
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),  # -> ceil(H/4)
                nn.ReLU(),
            )
            # infer conv output
            with torch.no_grad():
                x = torch.zeros((1, C, H, W))
                y = self.conv(x)
            self._conv_out_shape = y.shape[1:]  # (hidden, h2, w2)
            flat = int(np.prod(self._conv_out_shape))
            self.fc = nn.Linear(flat, embed_dim)
        elif len(self.obs_shape) == 1:
            D = int(self.obs_shape[0])
            self.conv = None
            self.fc = nn.Sequential(
                nn.Linear(D, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, embed_dim),
            )
        else:
            raise ValueError(f"Unsupported obs_shape={self.obs_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return self.fc(x)
        y = self.conv(x)
        y = y.reshape(y.size(0), -1)
        return self.fc(y)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int, embed_dim: int, obs_shape: Tuple[int, ...], conv_out_shape: Optional[Tuple[int, ...]]):
        super().__init__()
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.embed_dim = int(embed_dim)
        self.obs_shape = tuple(obs_shape)

        if len(self.obs_shape) == 3:
            C, H, W = self.obs_shape
            assert conv_out_shape is not None
            hc, h2, w2 = conv_out_shape
            flat = int(hc * h2 * w2)

            self.fc = nn.Linear(embed_dim, flat)
            # two deconvs to return to HxW
            # choose output_padding for second layer to match exact size for common small grids (e.g., 10x10)
            self.deconv1 = nn.ConvTranspose2d(hc, hidden_channels, 3, stride=2, padding=1, output_padding=0)
            self.deconv2 = nn.ConvTranspose2d(hidden_channels, C, 3, stride=2, padding=1, output_padding=1)

        elif len(self.obs_shape) == 1:
            D = int(self.obs_shape[0])
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, D),
            )
            self.deconv1 = None
            self.deconv2 = None
        else:
            raise ValueError(f"Unsupported obs_shape={self.obs_shape}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.deconv1 is None:
            return self.fc(z)
        # image-like
        C, H, W = self.obs_shape
        y = self.fc(z)
        # infer conv_out from fc weight
        # (B, flat) -> (B, hc, h2, w2)
        # We can recover hc,h2,w2 from deconv1 input channels and fc output size.
        hc = int(self.deconv1.in_channels)
        flat = y.size(1)
        h2w2 = flat // hc
        # assume square-ish h2,w2 by reading from output_padding design is fragile; instead keep from construction:
        # easiest: store a reshape shape in module state; but we already know it via weight sizes:
        # We'll store it by attaching attribute at init if you want; here keep it simple by using view with -1 for spatial.
        # For safety: use deconv1 weight expectations; but ok for our fixed construction.
        # We attach a hidden attribute if present:
        if hasattr(self, "_conv_out_shape"):
            hc2, h2, w2 = getattr(self, "_conv_out_shape")
        else:
            # fallback: assume h2 == w2 == int(sqrt(h2w2))
            h2 = int(np.sqrt(h2w2))
            w2 = h2
        y = y.view(y.size(0), hc, h2, w2)
        y = F.relu(self.deconv1(y))
        y = self.deconv2(y)
        return y


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        obs_shape: Tuple[int, ...],
        codebook_size: int = 512,
        embed_dim: int = 32,
        hidden_channels: int = 64,
        beta: float = 0.25,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.obs_shape = tuple(obs_shape)
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.hidden_channels = int(hidden_channels)
        self.beta = float(beta)

        self.encoder = Encoder(in_channels, hidden_channels, embed_dim, self.obs_shape)

        conv_out_shape = getattr(self.encoder, "_conv_out_shape", None)
        self.decoder = Decoder(in_channels, hidden_channels, embed_dim, self.obs_shape, conv_out_shape)
        # persist conv_out_shape for reshape safety
        if conv_out_shape is not None:
            setattr(self.decoder, "_conv_out_shape", conv_out_shape)

        self.vq = VectorQuantizer(codebook_size, embed_dim, beta=beta)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encoder(x)                       # (B,D)
        z_q, idx, vq_loss = self.vq(z_e)            # (B,D),(B,),()
        x_hat = self.decoder(z_q)                   # same shape as x (ideally)
        return dict(x_hat=x_hat, idx=idx, vq_loss=vq_loss, z_e=z_e)

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        # compute nearest codebook index
        z = out
        z2 = (z ** 2).sum(dim=1, keepdim=True)
        e2 = (self.vq.emb.weight ** 2).sum(dim=1).unsqueeze(0)
        ze = z @ self.vq.emb.weight.t()
        dist = z2 + e2 - 2.0 * ze
        idx = torch.argmin(dist, dim=1)
        return idx
