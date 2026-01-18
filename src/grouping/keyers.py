from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.models.vqvae import VQVAE


class GroupKeyer(ABC):
    """
    Maps an observation -> an integer group id.

    IMPORTANT: For theory-faithful runs, this mapping should be fixed over training.
    """

    @abstractmethod
    def __call__(self, obs: Any) -> int:
        raise NotImplementedError


class DiscreteIdentityKeyer(GroupKeyer):
    """
    For discrete scalar observations: group_id = int(obs).
    For non-scalar: hash bytes (stable for exact repeats only).
    """

    def __call__(self, obs: Any) -> int:
        arr = np.asarray(obs)
        if arr.ndim == 0:
            return int(arr.reshape(()))
        # Fallback: stable hash of bytes
        return int(np.uint64(hash(arr.tobytes())))


@dataclass
class SimHashKeyer(GroupKeyer):
    """
    Locality-sensitive hashing via random projections (SimHash style).

    This is inspired by count-based exploration uses of SimHash for high-dim observations,
    but here we use it as a coarse state aggregation keyer.

    - flattens obs -> x in R^d
    - computes bits = sign(R x), R ~ N(0,1)^{n_bits x d}
    - packs bits into an integer group id
    """

    n_bits: int = 16
    seed: int = 0
    _proj: Optional[np.ndarray] = None
    _d: Optional[int] = None

    def _ensure_proj(self, d: int):
        if self._proj is not None and self._d == d:
            return
        rng = np.random.default_rng(self.seed)
        self._proj = rng.standard_normal(size=(self.n_bits, d)).astype(np.float32)
        self._d = d

    def __call__(self, obs: Any) -> int:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        d = int(x.shape[0])
        self._ensure_proj(d)
        proj = self._proj  # (n_bits, d)
        bits = (proj @ x) >= 0.0  # (n_bits,)
        # pack bits -> int
        out = 0
        for i, b in enumerate(bits.tolist()):
            if b:
                out |= (1 << i)
        return int(out)


class VQVAEKeyer(GroupKeyer):
    """
    Uses a pretrained VQ-VAE encoder to map obs -> code index in {0..K-1}.

    Designed for small image-like inputs (e.g., MinAtar 10x10xC, your custom grids, etc).
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        cache_size: int = 200_000,
    ):
        self.device = str(device)
        self.cache_size = int(cache_size)
        self._cache: Dict[bytes, int] = {}

        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt["vqvae_cfg"]
        obs_shape = tuple(ckpt["obs_shape"])
        in_channels = int(cfg["in_channels"])

        self.model = VQVAE(
            in_channels=in_channels,
            obs_shape=obs_shape,
            codebook_size=int(cfg["codebook_size"]),
            embed_dim=int(cfg["embed_dim"]),
            hidden_channels=int(cfg["hidden_channels"]),
            beta=float(cfg["beta"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        x = np.asarray(obs)
        # Expect HWC (most gym envs), convert -> BCHW
        if x.ndim == 3:
            x = x.astype(np.float32)
            x = np.transpose(x, (2, 0, 1))  # CHW
            t = torch.from_numpy(x).unsqueeze(0)  # BCHW
            return t.to(self.device)
        # Vector-like: (D,)
        if x.ndim == 1:
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1, D)
            return t.to(self.device)
        # Scalar
        if x.ndim == 0:
            t = torch.tensor([float(x.reshape(()))], dtype=torch.float32, device=self.device).unsqueeze(0)
            return t
        raise ValueError(f"VQVAEKeyer got unsupported obs shape {x.shape}")

    def __call__(self, obs: Any) -> int:
        arr = np.asarray(obs)
        key = arr.tobytes()
        if key in self._cache:
            return int(self._cache[key])

        with torch.no_grad():
            x = self._obs_to_tensor(obs)
            idx = self.model.encode_indices(x)  # (B,)
            code = int(idx.item())

        # cache with bounded size
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                # cheap eviction: clear entirely (LRU is possible, but this is robust/simple)
                self._cache.clear()
            self._cache[key] = code
        return code
