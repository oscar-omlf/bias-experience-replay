from omegaconf import OmegaConf
from .uniform import UniformReplayBuffer
from .per import PrioritizedReplayBuffer
from .base import ReplayBuffer


def make_replay(cfg, obs_space, device: str) -> ReplayBuffer:
    rcfg = cfg.agents.replay
    typ = rcfg.type.lower()

    # Determine obs storage shape
    if hasattr(obs_space, "n"): # Discrete
        obs_shape = obs_space.n # we store ints and adapt in agent
        obs_shape = int(obs_shape)
    else:
        obs_shape = tuple(obs_space.shape)

    grouping_cfg = getattr(cfg.agents.replay, "grouping", None)
    grouping = None
    if grouping_cfg is not None:
        # OmegaConf -> plain dict
        grouping = OmegaConf.to_container(grouping_cfg, resolve=True)

    if typ == "uniform":
        return UniformReplayBuffer(rcfg.capacity, obs_shape=obs_shape, device=device)
    elif typ == "per":
        return PrioritizedReplayBuffer(
            rcfg.capacity,
            obs_shape=obs_shape,
            alpha=rcfg.alpha,
            beta=rcfg.beta,
            beta_anneal_steps=rcfg.beta_anneal_steps,
            device=device,
            grouping=grouping,
        )
    else:
        raise ValueError(f"Unknown replay type: {typ}")
