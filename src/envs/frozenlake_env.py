from typing import Tuple, Callable
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

def _obs_adapter_factory(n_states: int) -> Callable:
    """
    Map raw FrozenLake discrete observations to one-hot vectors.
    - Scalar int or 0-d array -> (n_states,) one-hot
    - Vector of length n_states -> passthrough (assumed one-hot-like)
    """
    def adapt(obs):
        arr = np.asarray(obs)

        # Scalar (gym returns Python int or 0-d array)
        if arr.ndim == 0:
            idx = int(arr.reshape(()))
            out = np.zeros(n_states, dtype=np.float32)
            out[idx] = 1.0
            return out

        # Already a vector: if it matches n_states, pass through
        if arr.ndim == 1 and arr.shape[0] == n_states:
            return arr.astype(np.float32)

        # Anything else is unexpected for per-sample adapter usage
        raise ValueError(
            f"FrozenLake adapter received shape {arr.shape}; expected scalar id or one-hot of length {n_states}."
        )
    return adapt


def make_frozenlake(cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    kwargs = dict(is_slippery=bool(cfg.is_slippery))

    if getattr(cfg, "success_rate", None) is not None:
        kwargs["success_rate"] = float(cfg.success_rate)

    if getattr(cfg, "reward_schedule", None) is not None:
        rs = cfg.reward_schedule
        # Hydra may provide ListConfig; convert to plain tuple
        kwargs["reward_schedule"] = (float(rs[0]), float(rs[1]), float(rs[2]))

    if getattr(cfg, "map_name", None):
        kwargs["map_name"] = cfg.map_name
    if getattr(cfg, "desc", None) is not None:
        kwargs["desc"] = cfg.desc
    if getattr(cfg, "render_mode", None):
        kwargs["render_mode"] = cfg.render_mode

    env = gym.make(cfg.id, max_episode_steps=int(cfg.max_episode_steps), **kwargs)
    eval_env = gym.make(cfg.id, max_episode_steps=int(cfg.max_episode_steps), **kwargs)

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
    eval_env = TimeLimit(eval_env, max_episode_steps=cfg.max_episode_steps)

    # Seed env RNGs and action spaces
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    n_states = int(env.unwrapped.observation_space.n)
    obs_adapter = _obs_adapter_factory(n_states)

    print(f"[Env] train obs_space={env.observation_space}, eval obs_space={eval_env.observation_space}")
    print(f"[Env] action_space={env.action_space}")
    print(f"[Env] success_rate={kwargs.get('success_rate', None)}, reward_schedule={kwargs.get('reward_schedule', None)}")

    return env, eval_env, obs_adapter
