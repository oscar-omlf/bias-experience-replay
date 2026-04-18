from typing import Any, Optional
import numpy as np

from .uniform import UniformReplayBuffer
from .per import PrioritizedReplayBuffer
from .base import ReplayBuffer
from src.grouping.keyers import DiscreteIdentityKeyer, SimHashKeyer, VQVAEKeyer


def _build_group_keyer(cfg, obs_space, device: str) -> Optional[Any]:
    """
    Returns a GroupKeyer or None.
    Expected config:
      agents.replay.grouping.enabled: bool
      agents.replay.grouping.type: "discrete" | "hash" | "vqvae"
      agents.replay.grouping.*: params
    """
    grouping = getattr(cfg.agents.replay, "grouping", None)
    if grouping is None or (not bool(getattr(grouping, "enabled", False))):
        return None

    gtype = str(getattr(grouping, "type", "discrete")).lower()

    if gtype == "discrete":
        return DiscreteIdentityKeyer()

    if gtype == "hash":
        n_bits = int(getattr(grouping, "n_bits", 16))
        seed = int(getattr(grouping, "seed", 0))
        return SimHashKeyer(n_bits=n_bits, seed=seed)

    if gtype == "vqvae":
        ckpt_path = str(getattr(grouping, "ckpt_path"))
        keyer_device = str(getattr(grouping, "device", "cpu"))
        cache_size = int(getattr(grouping, "cache_size", 200_000))
        return VQVAEKeyer(ckpt_path=ckpt_path, device=keyer_device, cache_size=cache_size)

    raise ValueError(f"Unknown grouping.type={gtype}")


def make_replay(cfg, obs_space, device: str) -> ReplayBuffer:
    rcfg = cfg.agents.replay
    typ = rcfg.type.lower()

    # Determine obs storage shape
    if hasattr(obs_space, "n"): # Discrete
        obs_shape = obs_space.n # we store ints and adapt in agent
        obs_shape = int(obs_shape)
    else:
        obs_shape = tuple(obs_space.shape)

    if typ == "uniform":
        return UniformReplayBuffer(rcfg.capacity, obs_shape=obs_shape, device=device)
    elif typ == "per":
        keyer = _build_group_keyer(cfg, obs_space, device)

        return PrioritizedReplayBuffer(
            rcfg.capacity,
            obs_shape=obs_shape,
            alpha=rcfg.alpha,
            beta=rcfg.beta,
            beta_anneal_steps=rcfg.beta_anneal_steps,
            device=device,
            keyer=keyer,
            normalize_is_weights=bool(getattr(rcfg, "normalize_is_weights", True)),
        )
    else:
        raise ValueError(f"Unknown replay type: {typ}")
