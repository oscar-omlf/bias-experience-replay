import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from minatar import gym as minatar_gym

minatar_gym.register_envs()

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
        orig_a = int(action)
        exec_a = orig_a
        sticky = False

        if self._last_a is not None and self._rng.rand() < self.zeta:
            exec_a = int(self._last_a)
            sticky = True

        obs, rew, terminated, truncated, info = self.env.step(exec_a)

        if info is None:
            info = {}
        else:
            info = dict(info)

        info["sticky"] = sticky
        info["orig_action"] = orig_a
        info["exec_action"] = exec_a

        self._last_a = exec_a
        return obs, rew, terminated, truncated, info


class StickyCatastrophe(gym.Wrapper):
    """
    If a step was sticky (action got replaced), then with prob p_cat add a large reward shock.

    Use shock < 0 for catastrophe tails.
    """
    def __init__(self, env, p_cat: float, shock: float, seed: int | None = None):
        super().__init__(env)
        self.p_cat = float(p_cat)
        self.shock = float(shock)  # e.g., -20.0 or -50.0
        self._rng = np.random.RandomState(seed)

        self.cat_count = 0  # optional diagnostics

    def reset(self, **kwargs):
        self.cat_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        if info is None:
            info = {}
        else:
            info = dict(info)

        do_cat = bool(info.get("sticky", False)) and (self._rng.rand() < self.p_cat)
        if do_cat:
            rew = float(rew) + self.shock
            info["catastrophe"] = True
            info["catastrophe_shock"] = float(self.shock)
            self.cat_count += 1
        else:
            info["catastrophe"] = False
            info["catastrophe_shock"] = 0.0

        return obs, float(rew), terminated, truncated, info


def _obs_adapter_minatar():
    def adapt(obs):
        arr = np.asarray(obs)
        # Expected from MinAtar Gymnasium: (H, W, C) with bool/0-1
        if arr.ndim != 3:
            raise ValueError(f"MinAtar obs must be 3D HWC; got shape {arr.shape}")
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        return arr  # CHW
    return adapt

def make_minatar(cfg, seed: int):
    env = gym.make(cfg.id, render_mode=cfg.render_mode)
    eval_env = gym.make(cfg.id, render_mode=cfg.render_mode)

    zeta = cfg.sticky_zeta
    if zeta > 0:
        env = StickyAction(env, zeta=zeta, seed=seed)
        eval_env = StickyAction(eval_env, zeta=zeta, seed=seed + 123)

    cat_cfg = getattr(cfg, "catastrophe", None)
    if cat_cfg is not None and bool(getattr(cat_cfg, "enabled", False)):
        p_cat = float(getattr(cat_cfg, "p_cat_given_sticky", 5e-4))
        shock = float(getattr(cat_cfg, "shock", -20.0))
        apply_to_eval = bool(getattr(cat_cfg, "apply_to_eval", True))

        env = StickyCatastrophe(env, p_cat=p_cat, shock=shock, seed=seed + 999)
        if apply_to_eval:
            eval_env = StickyCatastrophe(eval_env, p_cat=p_cat, shock=shock, seed=seed + 1000)

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    env = TimeLimit(env, max_episode_steps=int(cfg.max_episode_steps))
    eval_env = TimeLimit(eval_env, max_episode_steps=int(cfg.max_episode_steps))

    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    obs_adapter = _obs_adapter_minatar()

    print(f"[Env] MinAtar: train obs_space={env.observation_space}, eval obs_space={eval_env.observation_space}")
    print(f"[Env] MinAtar: action_space={env.action_space}")
    return env, eval_env, obs_adapter
