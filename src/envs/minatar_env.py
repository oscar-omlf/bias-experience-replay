# src/envs/minatar_env.py
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

        info = {} if info is None else dict(info)
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
        self.shock = float(shock)
        self._rng = np.random.RandomState(seed)
        self.cat_count = 0

    def reset(self, **kwargs):
        self.cat_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        info = {} if info is None else dict(info)

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


class MeanPreservingTailOnPositiveReward(gym.Wrapper):
    """
    Mean-preserving heavy-tail noise applied ONLY when reward > 0 (informative transitions).

    For base reward r>0:
      with prob p_bad:  r' = r - M
      else:            r' = r + c, where c = p_bad*M/(1-p_bad)

    So E[r'] = r, but distribution has a fat left tail that PER will oversample.
    """
    def __init__(
        self,
        env: gym.Env,
        p_bad: float = 0.02,
        M: float = 30.0,
        clip_abs: float | None = None,
        apply_to_eval: bool = False,  # handled in make_minatar
        seed: int | None = None,
    ):
        super().__init__(env)
        self.p_bad = float(p_bad)
        if not (0.0 < self.p_bad < 1.0):
            raise ValueError(f"p_bad must be in (0,1); got {self.p_bad}")
        self.M = float(M)
        if self.M <= 0:
            raise ValueError(f"M must be > 0; got {self.M}")

        self.c = (self.p_bad * self.M) / (1.0 - self.p_bad)
        self.clip_abs = None if clip_abs is None else float(clip_abs)
        self._rng = np.random.RandomState(seed)

        # diagnostics
        self.n_applied = 0
        self.n_bad = 0

    def reset(self, **kwargs):
        self.n_applied = 0
        self.n_bad = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        info = {} if info is None else dict(info)

        r = float(rew)
        # Apply only to informative rewards
        if r > 0.0:
            self.n_applied += 1
            u = self._rng.rand()
            if u < self.p_bad:
                r2 = r - self.M
                self.n_bad += 1
                info["mp_tail_applied"] = True
                info["mp_tail_bad"] = True
                info["mp_tail_delta"] = float(-self.M)
            else:
                r2 = r + self.c
                info["mp_tail_applied"] = True
                info["mp_tail_bad"] = False
                info["mp_tail_delta"] = float(self.c)

            if self.clip_abs is not None:
                r2 = float(np.clip(r2, -self.clip_abs, self.clip_abs))
            rew = r2
        else:
            info["mp_tail_applied"] = False
            info["mp_tail_bad"] = False
            info["mp_tail_delta"] = 0.0

        # optional always-on counters
        info["mp_tail_count_applied"] = int(self.n_applied)
        info["mp_tail_count_bad"] = int(self.n_bad)

        return obs, float(rew), terminated, truncated, info


def _obs_adapter_minatar():
    def adapt(obs):
        arr = np.asarray(obs)
        if arr.ndim != 3:
            raise ValueError(f"MinAtar obs must be 3D HWC; got shape {arr.shape}")
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        return arr  # CHW
    return adapt


def make_minatar(cfg, seed: int):
    env = gym.make(cfg.id, render_mode=cfg.render_mode)
    eval_env = gym.make(cfg.id, render_mode=cfg.render_mode)

    # Sticky actions first (affects dynamics / state distribution)
    zeta = float(cfg.sticky_zeta)
    if zeta > 0:
        env = StickyAction(env, zeta=zeta, seed=seed)
        eval_env = StickyAction(eval_env, zeta=zeta, seed=seed + 123)

    # ---- Catastrophe OR mean-preserving tail (mutually exclusive by default) ----
    cat_cfg = getattr(cfg, "catastrophe", None)
    cat_enabled = bool(getattr(cat_cfg, "enabled", False)) if cat_cfg is not None else False

    mp_cfg = getattr(cfg, "mp_tail", None)
    mp_enabled = bool(getattr(mp_cfg, "enabled", False)) if mp_cfg is not None else False

    if cat_enabled:
        p_cat = float(getattr(cat_cfg, "p_cat_given_sticky", 5e-4))
        shock = float(getattr(cat_cfg, "shock", -20.0))
        apply_to_eval = bool(getattr(cat_cfg, "apply_to_eval", True))

        env = StickyCatastrophe(env, p_cat=p_cat, shock=shock, seed=seed + 999)
        if apply_to_eval:
            eval_env = StickyCatastrophe(eval_env, p_cat=p_cat, shock=shock, seed=seed + 1000)

    elif mp_enabled:
        p_bad = float(getattr(mp_cfg, "p_bad", 0.02))
        M = float(getattr(mp_cfg, "M", 30.0))
        clip_abs = getattr(mp_cfg, "clip_abs", None)
        clip_abs = None if (clip_abs is None or str(clip_abs).lower() == "none") else float(clip_abs)
        apply_to_eval = bool(getattr(mp_cfg, "apply_to_eval", False))

        env = MeanPreservingTailOnPositiveReward(env, p_bad=p_bad, M=M, clip_abs=clip_abs, seed=seed + 2001)
        if apply_to_eval:
            eval_env = MeanPreservingTailOnPositiveReward(eval_env, p_bad=p_bad, M=M, clip_abs=clip_abs, seed=seed + 2002)

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
