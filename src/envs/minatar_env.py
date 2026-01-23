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
        self.shock = float(shock)
        self._rng = np.random.RandomState(seed)
        self.cat_count = 0

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
            rew = float(rew) + float(self.shock)
            info["catastrophe"] = True
            info["catastrophe_shock"] = float(self.shock)
            self.cat_count += 1
        else:
            info["catastrophe"] = False
            info["catastrophe_shock"] = 0.0

        info["catastrophe_count"] = int(self.cat_count)
        return obs, float(rew), terminated, truncated, info


class RewardHeavyTailNoise(gym.Wrapper):
    """
    Add *zero-mean, heavy-tailed* noise to reward.

    Default is a mixture-of-Gaussians:
      noise ~ { N(0, sigma_base) w.p. (1 - p_tail),
                N(0, sigma_tail) w.p. p_tail }

    This is mean-0, but produces rare large shocks.

    Options:
      - only_on_nonzero_reward: if True, only add noise when env reward != 0.
        (Often better for sparse-reward MinAtar so you don't drown learning in noise.)
      - clip_abs: if set, clips the added noise to [-clip_abs, +clip_abs].
    """
    def __init__(
        self,
        env: gym.Env,
        p_tail: float = 1e-3,
        sigma_base: float = 0.0,
        sigma_tail: float = 20.0,
        only_on_nonzero_reward: bool = True,
        clip_abs: float | None = None,
        seed: int | None = None,
    ):
        super().__init__(env)
        self.p_tail = float(p_tail)
        self.sigma_base = float(sigma_base)
        self.sigma_tail = float(sigma_tail)
        self.only_on_nonzero_reward = bool(only_on_nonzero_reward)
        self.clip_abs = None if clip_abs is None else float(clip_abs)
        self._rng = np.random.RandomState(seed)

        # diagnostics
        self.steps = 0
        self.tail_events = 0
        self.noise_sum = 0.0
        self.noise_abs_sum = 0.0

    def reset(self, **kwargs):
        self.steps = 0
        self.tail_events = 0
        self.noise_sum = 0.0
        self.noise_abs_sum = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        if info is None:
            info = {}
        else:
            info = dict(info)

        self.steps += 1

        r = float(rew)
        apply = True
        if self.only_on_nonzero_reward and (r == 0.0):
            apply = False

        noise = 0.0
        tail = False
        if apply:
            u = self._rng.rand()
            tail = (u < self.p_tail)
            sigma = self.sigma_tail if tail else self.sigma_base
            if sigma > 0.0:
                noise = float(self._rng.normal(loc=0.0, scale=sigma))
                if self.clip_abs is not None:
                    noise = float(np.clip(noise, -self.clip_abs, self.clip_abs))
            if tail:
                self.tail_events += 1

            r = r + noise
            self.noise_sum += noise
            self.noise_abs_sum += abs(noise)

        info["reward_noise/applied"] = bool(apply)
        info["reward_noise/tail"] = bool(tail)
        info["reward_noise/noise"] = float(noise)
        info["reward_noise/tail_events"] = int(self.tail_events)
        info["reward_noise/steps"] = int(self.steps)
        # (optional running averages)
        denom = float(max(1, self.steps))
        info["reward_noise/mean_noise"] = float(self.noise_sum / denom)
        info["reward_noise/mean_abs_noise"] = float(self.noise_abs_sum / denom)

        return obs, float(r), terminated, truncated, info


def _obs_adapter_minatar():
    def adapt(obs):
        arr = np.asarray(obs)
        if arr.ndim != 3:
            raise ValueError(f"MinAtar obs must be 3D HWC; got shape {arr.shape}")
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        return arr
    return adapt


def make_minatar(cfg, seed: int):
    env = gym.make(cfg.id, render_mode=cfg.render_mode)
    eval_env = gym.make(cfg.id, render_mode=cfg.render_mode)

    # Sticky actions
    zeta = float(getattr(cfg, "sticky_zeta", 0.0))
    if zeta > 0:
        env = StickyAction(env, zeta=zeta, seed=seed)
        eval_env = StickyAction(eval_env, zeta=zeta, seed=seed + 123)

    # Catastrophe vs heavy-tail noise (exclusive by default)
    cat_cfg = getattr(cfg, "catastrophe", None)
    noise_cfg = getattr(cfg, "reward_noise", None)

    cat_enabled = (cat_cfg is not None) and bool(getattr(cat_cfg, "enabled", False))
    noise_enabled = (noise_cfg is not None) and bool(getattr(noise_cfg, "enabled", False))

    if cat_enabled:
        p_cat = float(getattr(cat_cfg, "p_cat_given_sticky", 5e-4))
        shock = float(getattr(cat_cfg, "shock", -20.0))
        apply_to_eval = bool(getattr(cat_cfg, "apply_to_eval", True))

        env = StickyCatastrophe(env, p_cat=p_cat, shock=shock, seed=seed + 999)
        if apply_to_eval:
            eval_env = StickyCatastrophe(eval_env, p_cat=p_cat, shock=shock, seed=seed + 1000)

    elif noise_enabled:
        p_tail = float(getattr(noise_cfg, "p_tail", 1e-3))
        sigma_base = float(getattr(noise_cfg, "sigma_base", 0.0))
        sigma_tail = float(getattr(noise_cfg, "sigma_tail", 20.0))
        only_on_nonzero_reward = bool(getattr(noise_cfg, "only_on_nonzero_reward", True))
        clip_abs = getattr(noise_cfg, "clip_abs", None)
        apply_to_eval = bool(getattr(noise_cfg, "apply_to_eval", False))

        env = RewardHeavyTailNoise(
            env,
            p_tail=p_tail,
            sigma_base=sigma_base,
            sigma_tail=sigma_tail,
            only_on_nonzero_reward=only_on_nonzero_reward,
            clip_abs=clip_abs,
            seed=seed + 2000,
        )
        if apply_to_eval:
            eval_env = RewardHeavyTailNoise(
                eval_env,
                p_tail=p_tail,
                sigma_base=sigma_base,
                sigma_tail=sigma_tail,
                only_on_nonzero_reward=only_on_nonzero_reward,
                clip_abs=clip_abs,
                seed=seed + 3000,
            )

    # Standard wrappers
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
