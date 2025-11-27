from typing import Tuple, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from gymnasium.utils import seeding

from .frozenlake_env import _obs_adapter_factory


class ToyPERBiasEnv(gym.Env):
    """
    Toy MDP to study PER bias.

    States (Discrete(8)):
      0: S0 (start)
      1: S1 (safe chain)
      2: S2
      3: S3
      4: S4
      5: R  (risky state)
      6: G  (goal, terminal)
      7: H  (hole, terminal)

    Actions (Discrete(2)):
      - At S0: 0 = safe, 1 = risky
      - Elsewhere: action is ignored (only one effective action "cont")

    Dynamics:
      SAFE branch:
        S0 --(a=0, r=0)--> S1 --0--> S2 --0--> S3 --0--> S4 --(r=1)--> G

      RISKY branch:
        S0 --(a=1, r=0)--> R
        R --(p_success, r=r_high)--> G
        R --(1-p_success, r=0)-----> H
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
            self, 
            p_success: float = 0.1, r_high: float = 5.0, render_mode=None):
        super().__init__()
        self.p_success = float(p_success)
        self.r_high = float(r_high)

        self.render_mode = render_mode

        # States: S0, S1, S2, S3, S4, R, G, H
        self.observation_space = spaces.Discrete(8)
        # Actions: 0 = safe, 1 = risky (ignored except at S0)
        self.action_space = spaces.Discrete(2)

        self.state = 0
        self.np_random, _ = seeding.np_random(None)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self.state = 0  # S0
        info = {}
        return self.state, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        s = self.state
        terminated = False
        truncated = False
        reward = 0.0

        S0, S1, S2, S3, S4, R, G, H = range(8)

        # If already terminal, stay there (wrappers shouldn't call step after done,
        # but handle gracefully in case)
        if s in (G, H):
            return s, 0.0, True, False, {}

        if s == S0:
            # Choose safe vs risky
            if action == 0:
                ns = S1
                reward = 0.0
            else:
                ns = R
                reward = 0.0

        elif s in (S1, S2, S3):
            # Deterministic safe chain with delayed reward at the end
            ns = s + 1

        elif s == S4:
            ns = G
            reward = 1.0
            terminated = True

        elif s == R:
            # Stochastic risky outcome with big rare reward
            if self.np_random.random() < self.p_success:
                ns = G
                reward = self.r_high
            else:
                ns = H
                reward = 0.0
            terminated = True

        else:
            # Should not happen, but be safe
            ns = s

        if ns in (G, H):
            terminated = True

        self.state = ns
        info = {}
        return ns, float(reward), terminated, truncated, info

    def render(self):
        S0, S1, S2, S3, S4, R, G, H = range(8)
        names = {
            S0: "S0(start)",
            S1: "S1",
            S2: "S2",
            S3: "S3",
            S4: "S4",
            R:  "R(risky)",
            G:  "G(goal)",
            H:  "H(hole)",
        }
        print(f"State: {names.get(self.state, self.state)}")

    def close(self):
        pass


def make_toy_per_bias(cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    """
    Factory matching your make_frozenlake signature:
      returns (env, eval_env, obs_adapter)
    """
    env = ToyPERBiasEnv(
        p_success=getattr(cfg, "p_success", 0.1),
        r_high=getattr(cfg, "r_high", 5.0),
        render_mode=getattr(cfg, "render_mode", None),
    )
    eval_env = ToyPERBiasEnv(
        p_success=getattr(cfg, "p_success", 0.1),
        r_high=getattr(cfg, "r_high", 5.0),
        render_mode=getattr(cfg, "render_mode", None),
    )

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    max_steps = int(getattr(cfg, "max_episode_steps", 20))
    env = TimeLimit(env, max_episode_steps=max_steps)
    eval_env = TimeLimit(eval_env, max_episode_steps=max_steps)

    # Seed envs and action spaces
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    n_states = int(env.unwrapped.observation_space.n)
    obs_adapter = _obs_adapter_factory(n_states)

    print(f"[Env] ToyPERBias train obs_space={env.observation_space}, eval obs_space={eval_env.observation_space}")
    print(f"[Env] ToyPERBias action_space={env.action_space}")

    return env, eval_env, obs_adapter
