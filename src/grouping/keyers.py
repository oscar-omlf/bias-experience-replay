from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time

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

    Robust to obs shaped:
      - HWC (gym default for MinAtar)
      - CHW (if you ever store adapted obs)
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        cache_size: int = 200_000,
    ):
        self.device = str(device)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cache_size = int(cache_size)
        self._cache: Dict[bytes, int] = {}

        # Stats
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.encode_time_ms = 0.0  # cumulative

        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt["vqvae_cfg"]
        self.obs_shape = tuple(ckpt["obs_shape"])  # (C,H,W)
        self.in_channels = int(cfg["in_channels"])
        self.codebook_size = int(cfg["codebook_size"])

        self.model = VQVAE(
            in_channels=self.in_channels,
            obs_shape=self.obs_shape,
            codebook_size=self.codebook_size,
            embed_dim=int(cfg["embed_dim"]),
            hidden_channels=int(cfg["hidden_channels"]),
            beta=float(cfg["beta"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def _obs_to_bchw(self, obs: Any) -> torch.Tensor:
        x = np.asarray(obs, dtype=np.float32)

        if x.ndim != 3:
            raise ValueError(f"VQVAEKeyer expects 3D obs; got {x.shape}")

        C, H, W = self.obs_shape

        if x.shape == (C, H, W):
            chw = x
        elif x.shape == (H, W, C):
            chw = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(
                f"Obs shape mismatch for VQ-VAE: got {x.shape}, expected CHW {(C,H,W)} or HWC {(H,W,C)}"
            )

        chw = np.ascontiguousarray(chw, dtype=np.float32)
        t = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        return t

    def stats(self) -> Dict[str, float]:
        calls = max(1, int(self.calls))
        hit_rate = float(self.cache_hits) / float(calls)
        avg_ms = float(self.encode_time_ms) / float(max(1, int(self.cache_misses)))
        return {
            "vqvae_codebook_size": float(self.codebook_size),
            "cache_size": float(len(self._cache)),
            "cache_hits": float(self.cache_hits),
            "cache_misses": float(self.cache_misses),
            "cache_hit_rate": float(hit_rate),
            "encode_ms_per_miss": float(avg_ms),
        }

    def __call__(self, obs: Any) -> int:
        self.calls += 1

        arr = np.asarray(obs)
        key = arr.tobytes()

        cached = self._cache.get(key, None)
        if cached is not None:
            self.cache_hits += 1
            return int(cached)

        self.cache_misses += 1
        t0 = time.time()
        with torch.no_grad():
            x = self._obs_to_bchw(obs)
            idx = self.model.encode_indices(x)  # (1,)
            code = int(idx.item())
        self.encode_time_ms += 1000.0 * (time.time() - t0)

        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.clear()
            self._cache[key] = code

        return code
