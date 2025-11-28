from typing import Tuple, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from gymnasium.utils import seeding

from .frozenlake_env import _obs_adapter_factory


class ConalBanditEnv(gym.Env):
    """
    Conal multi-armed bandit from 'Uncertainty Prioritized Experience Replay' (https://arxiv.org/abs/2506.09270).

    States:
      - Single dummy state s=0. The problem is stateless (k-arm bandit).

    Actions:
      - A = {0, 1, ..., n_arms-1}, each action is an arm.

    Reward:
      - For chosen arm a:
          r(a) = r_bar + eta * sigma(a)
          eta ~ Normal(0, 1)
          sigma(a) = a/(n_arms-1) * sigma_max + sigma_min

      - All arms have the same mean r_bar, increasing noise as a increases.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        n_arms: int = 5,
        r_bar: float = 2.0,
        sigma_max: float = 2.0,
        sigma_min: float = 0.1,
        render_mode=None,
    ):
        super().__init__()

        self.n_arms = int(n_arms)
        self.r_bar = float(r_bar)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.render_mode = render_mode

        # One dummy state
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(self.n_arms)

        self.state = 0
        self.np_random, _ = seeding.np_random(None)

    def _sigma(self, a: int) -> float:
        # a in [0, n_arms-1]
        if self.n_arms == 1:
            return self.sigma_min
        frac = float(a) / float(self.n_arms - 1)
        return frac * self.sigma_max + self.sigma_min

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        sigma_a = self._sigma(int(action))
        eta = self.np_random.normal(loc=0.0, scale=1.0)
        reward = self.r_bar + eta * sigma_a

        # Stateless continuing task: never 'terminated' inside env.
        terminated = False
        truncated = False
        info = {}

        return self.state, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return "ConalBanditEnv(state=0)"
        elif self.render_mode == "human":
            print("ConalBanditEnv: single-state bandit.")
        else:
            return None

    def close(self):
        pass


def make_conal_bandit(cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    """
    cfg has:
      - n_arms
      - r_bar
      - sigma_max
      - sigma_min
      - max_episode_steps
      - render_mode
    """
    kwargs = dict(
        n_arms=int(getattr(cfg, "n_arms", 5)),
        r_bar=float(getattr(cfg, "r_bar", 2.0)),
        sigma_max=float(getattr(cfg, "sigma_max", 2.0)),
        sigma_min=float(getattr(cfg, "sigma_min", 0.1)),
        render_mode=getattr(cfg, "render_mode", None),
    )

    env = ConalBanditEnv(**kwargs)
    eval_env = ConalBanditEnv(**kwargs)

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    max_steps = int(getattr(cfg, "max_episode_steps", 1000))
    env = TimeLimit(env, max_episode_steps=max_steps)
    eval_env = TimeLimit(eval_env, max_episode_steps=max_steps)

    # Seed envs and action spaces
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    # Dummy state because htis is a k-arm bandit
    n_states = int(env.unwrapped.observation_space.n)
    obs_adapter = _obs_adapter_factory(n_states)

    print(f"[Env] ConalBandit train obs_space={env.observation_space}, eval obs_space={eval_env.observation_space}")
    print(f"[Env] ConalBandit action_space={env.action_space}")

    return env, eval_env, obs_adapter
