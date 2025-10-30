from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

import minatar

class StickyAction(gym.Wrapper):
    def __init__(self, env, zeta: float = 0.25, seed: int | None = None):
        super().__init__(env)
        self.zeta = float(zeta)
        self._rng = np.random.RandomState(seed)
        self._last_a = None

    def reset(self, **kwargs):
        self._last_a = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._last_a is not None and self._rng.rand() < self.zeta:
            action = self._last_a
        obs, rew, terminated, truncated, info = self.env.step(action)
        self._last_a = action
        return obs, rew, terminated, truncated, info

def _obs_adapter_miniatar():
    def adapt(obs):
        arr = np.asarray(obs)
        # Expected from MinAtar Gymnasium: (H, W, C) with bool/0-1
        if arr.ndim != 3:
            raise ValueError(f"MiniAtar obs must be 3D HWC; got shape {arr.shape}")
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        return arr  # CHW
    return adapt

def make_miniatar(cfg, seed: int):
    env = gym.make(cfg.id, render_mode=cfg.render_mode)
    eval_env = gym.make(cfg.id, render_mode=cfg.render_mode)

    zeta = cfg.sticky_zeta
    if zeta > 0:
        env = StickyAction(env, zeta=zeta, seed=seed)
        eval_env = StickyAction(eval_env, zeta=zeta, seed=seed + 123)

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    env = TimeLimit(env, max_episode_steps=int(cfg.max_episode_steps))
    eval_env = TimeLimit(eval_env, max_episode_steps=int(cfg.max_episode_steps))

    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    obs_adapter = _obs_adapter_miniatar()

    print(f"[Env] MiniAtar: train obs_space={env.observation_space}, eval obs_space={eval_env.observation_space}")
    print(f"[Env] MiniAtar: action_space={env.action_space}")
    return env, eval_env, obs_adapter
