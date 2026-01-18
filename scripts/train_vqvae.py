import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

import gymnasium as gym

from src.models.vqvae import VQVAE
from src.utils.seed import set_global_seeds


@dataclass
class VQVAETrainCfg:
    # data collection
    env_id: str = "MinAtar/Breakout-v0"
    seed: int = 0
    collect_steps: int = 200_000

    # model
    codebook_size: int = 512
    embed_dim: int = 32
    hidden_channels: int = 64
    beta: float = 0.25

    # training
    batch_size: int = 256
    train_steps: int = 50_000
    lr: float = 3e-4
    recon_loss: str = "bce"  # "mse" or "bce"

    # output
    save_path: str = "artifacts/vqvae.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor_batch(obs_batch: np.ndarray, device: str) -> torch.Tensor:
    x = np.asarray(obs_batch)
    if x.ndim == 4:
        # (B,H,W,C) -> (B,C,H,W)
        x = x.astype(np.float32)
        x = np.transpose(x, (0, 3, 1, 2))
        return torch.from_numpy(x).to(device)
    if x.ndim == 2:
        return torch.from_numpy(x.astype(np.float32)).to(device)
    raise ValueError(f"Unsupported batch shape {x.shape}")


@hydra.main(config_path="../config", config_name="train_vqvae", version_base=None)
def main(cfg: DictConfig):
    cfg_py = OmegaConf.to_container(cfg, resolve=True)
    cfg2 = VQVAETrainCfg(**cfg_py)

    os.makedirs(os.path.dirname(cfg2.save_path), exist_ok=True)
    set_global_seeds(cfg2.seed)

    env = gym.make(cfg2.env_id)
    obs, _ = env.reset(seed=cfg2.seed)

    # collect observations
    obs_list = []
    for t in range(int(cfg2.collect_steps)):
        a = env.action_space.sample()
        next_obs, _r, terminated, truncated, _info = env.step(a)
        obs_list.append(obs)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    obs_arr = np.asarray(obs_list)
    # determine shape
    if obs_arr.ndim == 4:
        # (N,H,W,C)
        H, W, C = obs_arr.shape[1], obs_arr.shape[2], obs_arr.shape[3]
        obs_shape = (C, H, W)
        in_channels = C
    elif obs_arr.ndim == 2:
        D = obs_arr.shape[1]
        obs_shape = (D,)
        in_channels = 1
    else:
        raise ValueError(f"Unexpected collected obs shape: {obs_arr.shape}")

    device = cfg2.device
    model = VQVAE(
        in_channels=in_channels,
        obs_shape=obs_shape,
        codebook_size=cfg2.codebook_size,
        embed_dim=cfg2.embed_dim,
        hidden_channels=cfg2.hidden_channels,
        beta=cfg2.beta,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg2.lr)

    # training loop
    n = obs_arr.shape[0]
    bs = int(cfg2.batch_size)

    for step in range(int(cfg2.train_steps)):
        idx = np.random.randint(0, n, size=bs)
        batch = obs_arr[idx]

        x = _to_tensor_batch(batch, device=device)

        out = model(x)
        x_hat = out["x_hat"]
        vq_loss = out["vq_loss"]

        if cfg2.recon_loss == "bce":
            # Assume obs is binary-ish in [0,1] for MinAtar; clamp for numerical safety
            x_in = x.clamp(0.0, 1.0)
            xh = torch.sigmoid(x_hat)
            recon = torch.mean(F.binary_cross_entropy(xh, x_in, reduction="none"))
        else:
            recon = torch.mean((x_hat - x) ** 2)

        loss = recon + vq_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(f"[vqvae] step={step} loss={float(loss.item()):.6f} recon={float(recon.item()):.6f} vq={float(vq_loss.item()):.6f}")

    ckpt = {
        "state_dict": model.state_dict(),
        "obs_shape": obs_shape,
        "vqvae_cfg": {
            "in_channels": in_channels,
            "obs_shape": obs_shape,
            "codebook_size": cfg2.codebook_size,
            "embed_dim": cfg2.embed_dim,
            "hidden_channels": cfg2.hidden_channels,
            "beta": cfg2.beta,
        },
    }
    torch.save(ckpt, cfg2.save_path)
    print(f"[vqvae] saved -> {cfg2.save_path}")


if __name__ == "__main__":
    main()
