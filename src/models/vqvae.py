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
        Supports:
          - z_e: (B, D)                      -> idx (B,)
          - z_e: (B, D, Gh, Gw)              -> idx (B, Gh, Gw)

        Returns:
          z_q_st: same shape as z_e (straight-through)
          idx: (B,) or (B,Gh,Gw)
          vq_loss: scalar
        """
        if z_e.ndim == 2:
            z = z_e  # (B,D)
            B, D = z.shape
            z_flat = z
            reshape_idx = ("vec", B)
        elif z_e.ndim == 4:
            # (B,D,Gh,Gw) -> (B*Gh*Gw, D)
            B, D, Gh, Gw = z_e.shape
            z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)
            reshape_idx = ("grid", B, Gh, Gw)
        else:
            raise ValueError(f"VectorQuantizer expected z_e.ndim in {{2,4}}, got {z_e.ndim} with shape {tuple(z_e.shape)}")

        # distances: ||z||^2 + ||e||^2 - 2 z*e
        z2 = (z_flat ** 2).sum(dim=1, keepdim=True)               # (N,1)
        e2 = (self.emb.weight ** 2).sum(dim=1).unsqueeze(0)       # (1,K)
        ze = z_flat @ self.emb.weight.t()                         # (N,K)
        dist = z2 + e2 - 2.0 * ze                                 # (N,K)

        idx_flat = torch.argmin(dist, dim=1)                      # (N,)
        z_q_flat = self.emb(idx_flat)                             # (N,D)

        # losses
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        commit_loss = F.mse_loss(z_flat, z_q_flat.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through estimator
        z_q_st_flat = z_flat + (z_q_flat - z_flat).detach()

        # reshape back
        if reshape_idx[0] == "vec":
            z_q_st = z_q_st_flat
            idx = idx_flat
        else:
            _, B, Gh, Gw = reshape_idx
            z_q_st = z_q_st_flat.view(B, Gh, Gw, D).permute(0, 3, 1, 2).contiguous()   # (B,D,Gh,Gw)
            idx = idx_flat.view(B, Gh, Gw)                                             # (B,Gh,Gw)

        return z_q_st, idx, vq_loss


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embed_dim: int,
        obs_shape: Tuple[int, ...],
        grid_size: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.embed_dim = int(embed_dim)
        self.obs_shape = tuple(obs_shape)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))

        if len(self.obs_shape) == 3:
            C, H, W = self.obs_shape
            self.conv = nn.Sequential(
                nn.Conv2d(C, hidden_channels, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
                nn.ReLU(),
            )

            # For vector mode: flatten -> fc
            if self.grid_size == (1, 1):
                with torch.no_grad():
                    x = torch.zeros((1, C, H, W))
                    y = self.conv(x)
                self._conv_out_shape = y.shape[1:]  # (hidden, h2, w2)
                flat = int(np.prod(self._conv_out_shape))
                self.fc = nn.Linear(flat, embed_dim)
                self.pool = None
                self.proj = None
            else:
                # For grid mode: adaptive pool to (Gh,Gw) then 1x1 conv -> embed_dim
                self.fc = None
                self.pool = nn.AdaptiveAvgPool2d(self.grid_size)
                self.proj = nn.Conv2d(hidden_channels, embed_dim, kernel_size=1, stride=1, padding=0)

        elif len(self.obs_shape) == 1:
            if self.grid_size != (1, 1):
                raise ValueError("grid_size > 1 is only supported for image-like obs (CHW).")
            D = int(self.obs_shape[0])
            self.conv = None
            self.fc = nn.Sequential(
                nn.Linear(D, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, embed_dim),
            )
            self.pool = None
            self.proj = None
        else:
            raise ValueError(f"Unsupported obs_shape={self.obs_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
          - vector mode: (B, D)
          - grid mode:   (B, D, Gh, Gw)
        """
        if self.conv is None:
            return self.fc(x)

        y = self.conv(x)

        if self.grid_size == (1, 1):
            y = y.reshape(y.size(0), -1)
            return self.fc(y)

        y = self.pool(y)        # (B, hidden, Gh, Gw)
        z = self.proj(y)        # (B, embed_dim, Gh, Gw)
        return z


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        embed_dim: int,
        obs_shape: Tuple[int, ...],
        grid_size: Tuple[int, int] = (1, 1),
        conv_out_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.embed_dim = int(embed_dim)
        self.obs_shape = tuple(obs_shape)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))

        if len(self.obs_shape) == 3:
            C, H, W = self.obs_shape

            if self.grid_size == (1, 1):
                # original-ish decoder (vector -> convtranspose) so 1-code mode stays strong
                assert conv_out_shape is not None, "conv_out_shape required for vector decoder"
                hc, h2, w2 = conv_out_shape
                flat = int(hc * h2 * w2)

                self.fc = nn.Linear(embed_dim, flat)
                self.deconv1 = nn.ConvTranspose2d(hc, hidden_channels, 3, stride=2, padding=1, output_padding=0)
                self.deconv2 = nn.ConvTranspose2d(hidden_channels, C, 3, stride=2, padding=1, output_padding=1)
                self._conv_out_shape = (hc, h2, w2)

                self.grid_proj = None
                self.grid_conv = None
            else:
                # grid decoder: (B,D,Gh,Gw) -> upsample -> convs -> (B,C,H,W)
                self.fc = None
                self.deconv1 = None
                self.deconv2 = None

                self.grid_proj = nn.Conv2d(embed_dim, hidden_channels, kernel_size=1, stride=1, padding=0)
                self.grid_conv = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_channels, C, 1),
                )

        elif len(self.obs_shape) == 1:
            if self.grid_size != (1, 1):
                raise ValueError("grid_size > 1 is only supported for image-like obs (CHW).")
            D = int(self.obs_shape[0])
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, D),
            )
            self.deconv1 = None
            self.deconv2 = None
            self.grid_proj = None
            self.grid_conv = None
        else:
            raise ValueError(f"Unsupported obs_shape={self.obs_shape}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:
          - vector mode: (B, D)
          - grid mode:   (B, D, Gh, Gw)
        returns:
          - logits x_hat with shape matching obs (B,C,H,W) or (B,D)
        """
        if len(self.obs_shape) == 1:
            return self.fc(z)

        C, H, W = self.obs_shape

        # vector mode
        if self.grid_size == (1, 1):
            if z.ndim != 2:
                raise ValueError(f"Vector decoder expected z.ndim==2, got {z.ndim} with shape {tuple(z.shape)}")
            y = self.fc(z)
            hc, h2, w2 = self._conv_out_shape
            y = y.view(y.size(0), hc, h2, w2)
            y = F.relu(self.deconv1(y))
            y = self.deconv2(y)
            return y

        # grid mode
        if z.ndim != 4:
            raise ValueError(f"Grid decoder expected z.ndim==4, got {z.ndim} with shape {tuple(z.shape)}")

        y = self.grid_proj(z)                                    # (B, hidden, Gh, Gw)
        y = F.interpolate(y, size=(H, W), mode="nearest")         # (B, hidden, H, W)
        y = self.grid_conv(y)                                    # (B, C, H, W)
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
        grid_size: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.obs_shape = tuple(obs_shape)
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.hidden_channels = int(hidden_channels)
        self.beta = float(beta)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))

        self.encoder = Encoder(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            embed_dim=self.embed_dim,
            obs_shape=self.obs_shape,
            grid_size=self.grid_size,
        )

        conv_out_shape = getattr(self.encoder, "_conv_out_shape", None)
        self.decoder = Decoder(
            out_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            embed_dim=self.embed_dim,
            obs_shape=self.obs_shape,
            grid_size=self.grid_size,
            conv_out_shape=conv_out_shape,
        )

        self.vq = VectorQuantizer(self.codebook_size, self.embed_dim, beta=self.beta)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encoder(x)                    # (B,D) or (B,D,Gh,Gw)
        z_q, idx, vq_loss = self.vq(z_e)         # same shape as z_e, idx matches mode
        x_hat = self.decoder(z_q)                # (B,C,H,W) logits
        return dict(x_hat=x_hat, idx=idx, vq_loss=vq_loss, z_e=z_e)

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)                    # (B,D) or (B,D,Gh,Gw)
        _, idx, _ = self.vq(z_e)
        return idx
        
