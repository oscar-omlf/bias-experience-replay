import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from typing import Tuple, Callable

from .frozenlake_env import make_frozenlake
from .miniatar_env import make_miniatar
from .toy_per_bias_env import make_toy_per_bias


def make_env(env_cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    """Factory for training and evaluation envs + observation adapter.
    Returns (env, eval_env, obs_adapter) where obs_adapter(obs) -> np.ndarray.
    """
    env_id = env_cfg.id
    if env_id == "FrozenLake-v1":
        env, eval_env, obs_adapter = make_frozenlake(env_cfg, seed)
    elif env_id.startswith("MinAtar/"):
        env, eval_env, obs_adapter = make_miniatar(env_cfg, seed)
    elif env_id == "ToyPERBias-v0":
        env, eval_env, obs_adapter = make_toy_per_bias(env_cfg, seed)
    else:
        raise ValueError(f"Unsupported env id: {env_id}")

    return env, eval_env, obs_adapter
