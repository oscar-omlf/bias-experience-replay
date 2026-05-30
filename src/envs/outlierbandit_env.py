from typing import Tuple, Callable, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from .frozenlake_env import _obs_adapter_factory


class OutlierBandit(gym.Env):
    """
    1-step bandit with stochastic rewards.
    Single state s=0, choose action a in {0..K-1}, receive reward:
      r = r_high with prob p_success[a], else r_low
    Episode terminates after 1 step.
    Exposes env.unwrapped.true_means for diagnostics.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_actions: int = 2,
        p_success: Optional[list[float]] = None,
        r_high: float = 10.0,
        r_low: float = 0.0,
        render_mode=None,
    ):
        super().__init__()
        self.n_actions = int(n_actions)
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(self.n_actions)

        if p_success is None:
            p_success = [0.1] + [0.0] * (self.n_actions - 1)
        self.p_success = np.asarray(p_success, dtype=np.float32)
        assert self.p_success.shape[0] == self.n_actions

        self.r_high = self._broadcast_reward_param(r_high, name="r_high")
        self.r_low = self._broadcast_reward_param(r_low, name="r_low")
        self.render_mode = render_mode

        # For the agent's bandit metrics (expects 1D array)
        self.true_means = (self.p_success * self.r_high) + ((1.0 - self.p_success) * self.r_low)

        self.np_random, _ = seeding.np_random(None)

    def _broadcast_reward_param(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return np.full((self.n_actions,), float(arr.reshape(())), dtype=np.float32)
        if arr.shape != (self.n_actions,):
            raise ValueError(f"{name} must be a scalar or length-{self.n_actions} list; got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        return 0, {}

    def step(self, action):
        assert self.action_space.contains(action)
        a = int(action)
        if self.np_random.random() < float(self.p_success[a]):
            r = self.r_high[a]
        else:
            r = self.r_low[a]
        terminated = True
        truncated = False
        return 0, float(r), terminated, truncated, {}


def make_outlierbandit(cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    env = OutlierBandit(
        n_actions=int(getattr(cfg, "n_actions", 2)),
        p_success=getattr(cfg, "p_success", None),
        r_high=getattr(cfg, "r_high", 10.0),
        r_low=getattr(cfg, "r_low", 0.0),
    )
    eval_env = OutlierBandit(
        n_actions=int(getattr(cfg, "n_actions", 2)),
        p_success=getattr(cfg, "p_success", None),
        r_high=getattr(cfg, "r_high", 10.0),
        r_low=getattr(cfg, "r_low", 0.0),
    )

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    # One-step episode, but keep wrappers consistent
    env = TimeLimit(env, max_episode_steps=1)
    eval_env = TimeLimit(eval_env, max_episode_steps=1)

    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    obs_adapter = _obs_adapter_factory(1)
    return env, eval_env, obs_adapter
