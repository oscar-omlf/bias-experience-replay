# noisy_gridworld_env.py
from typing import Tuple, Callable, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from gymnasium.utils import seeding

from .frozenlake_env import _obs_adapter_factory


class NoisyGridworldEnv(gym.Env):
    """
    Noisy Gridworld from "Uncertainty Prioritized Experience Replay" (https://arxiv.org/abs/2506.09270).

    Layout (7 rows x 8 cols); rows indexed top -> bottom, cols left -> right.

        Row 0: S . . . . N N N
        Row 1: # # # # . N N N
        Row 2: . . . . . N N N
        Row 3: . # # # # N N N
        Row 4: . . . . # N N N
        Row 5: # # # . # N N N
        Row 6: G . . . # N N N

    Legend:
      S : start state (blue in the paper)
      G : goal state (green)
      . : normal corridor (zero-mean, deterministic reward, typically 0)
      # : wall (impassable)
      N : noisy states (yellow in the paper) with random reward

    Actions:
      0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP   (same order as FrozenLake-v1)

    Rewards (configurable via constructor args):
      - Stepping on 'N' (noisy states):
            r ~ Normal(noise_mean, noise_std^2)
      - Stepping on 'G' (goal state):
            r = goal_reward (deterministic)
      - All other transitions:
            r = step_penalty   (default 0.0)

    Episode terminates when:
      - Agent reaches G (terminated=True), or
      - TimeLimit wrapper truncates the episode (truncated=True).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        noise_mean: float = 0.0,
        noise_std: float = np.sqrt(2.0),   # N(0, 2)
        goal_reward: float = 10.0,
        step_penalty: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self._layout: List[str] = [
            "S....NNN",  # Row 0
            "####.NNN",  # Row 1
            ".....NNN",  # Row 2
            ".####NNN",  # Row 3
            "....#NNN",  # Row 4
            "###.#NNN",  # Row 5
            "G...#NNN",  # Row 6
        ]

        self.height = len(self._layout)
        self.width = len(self._layout[0])
        assert all(len(row) == self.width for row in self._layout), "Non-rectangular layout"

        self.noise_mean = float(noise_mean)
        self.noise_std = float(noise_std)
        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.render_mode = render_mode

        # Locate S and G
        self.start_pos = self._find_symbol("S")
        self.goal_pos = self._find_symbol("G")

        # Precompute set of noisy cells (N)
        self.noisy_cells = {
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if self._layout[r][c] == "N"
        }

        # Discrete state = row * width + col
        n_states = self.height * self.width
        self.observation_space = spaces.Discrete(n_states)
        # Same 4-action ordering as FrozenLake
        self.action_space = spaces.Discrete(4)  # LEFT, DOWN, RIGHT, UP

        self.np_random, _ = seeding.np_random(None)
        self.agent_pos = self.start_pos

    def _find_symbol(self, sym: str) -> Tuple[int, int]:
        for r, row in enumerate(self._layout):
            for c, ch in enumerate(row):
                if ch == sym:
                    return (r, c)
        raise ValueError(f"Symbol {sym!r} not found in layout")

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * self.width + c

    def _state_to_pos(self, s: int) -> Tuple[int, int]:
        r = s // self.width
        c = s % self.width
        return (r, c)

    def _cell(self, pos: Tuple[int, int]) -> str:
        r, c = pos
        return self._layout[r][c]

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        return self._cell(pos) == "#"


    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self.agent_pos = self.start_pos
        obs = self._pos_to_state(self.agent_pos)
        info = {}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        r, c = self.agent_pos

        # Same action mapping as FrozenLake:
        # 0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP
        if action == 0:
            new_r, new_c = r, c - 1
        elif action == 1:
            new_r, new_c = r + 1, c
        elif action == 2:
            new_r, new_c = r, c + 1
        else:
            new_r, new_c = r - 1, c

        # Check bounds
        if not (0 <= new_r < self.height and 0 <= new_c < self.width):
            new_r, new_c = r, c  # bump into wall

        new_pos = (new_r, new_c)

        # If new cell is a wall, stay in place
        if self._is_wall(new_pos):
            new_pos = (r, c)

        cell = self._cell(new_pos)

        # Base reward
        reward = self.step_penalty
        terminated = False
        truncated = False

        # Noisy region
        if new_pos in self.noisy_cells:
            reward += float(self.np_random.normal(self.noise_mean, self.noise_std))

        # Goal
        if cell == "G":
            reward += self.goal_reward
            terminated = True

        self.agent_pos = new_pos
        obs = self._pos_to_state(self.agent_pos)
        info = {}

        return obs, float(reward), terminated, truncated, info


    def render(self):
        # ASCII rendering
        lines: List[str] = []
        for r in range(self.height):
            row_chars: List[str] = []
            for c in range(self.width):
                pos = (r, c)
                ch = self._layout[r][c]
                if pos == self.agent_pos:
                    row_chars.append("A")
                else:
                    row_chars.append(ch)
            lines.append("".join(row_chars))
        out = "\n".join(lines)

        if self.render_mode == "human":
            print(out)
        return out

    def close(self):
        pass


def make_noisy_gridworld(cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    """
    Factory to integrate with your existing env registry.

    cfg is expected to have (defaults follow paper):
      - noise_mean: float (default 0.0)
      - noise_std: float (default sqrt(2.0))
      - goal_reward: float (default 10.0)
      - step_penalty: float (default 0.0)
      - max_episode_steps: int (default 1000)
      - render_mode: Optional[str]
    """
    kwargs = dict(
        noise_mean=float(getattr(cfg, "noise_mean", 0.0)),
        noise_std=float(getattr(cfg, "noise_std", np.sqrt(2.0))),
        goal_reward=float(getattr(cfg, "goal_reward", 10.0)),
        step_penalty=float(getattr(cfg, "step_penalty", 0.0)),
        render_mode=getattr(cfg, "render_mode", None),
    )

    env = NoisyGridworldEnv(**kwargs)
    eval_env = NoisyGridworldEnv(**kwargs)

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    max_steps = int(getattr(cfg, "max_episode_steps", 1000))
    env = TimeLimit(env, max_episode_steps=max_steps)
    eval_env = TimeLimit(eval_env, max_episode_steps=max_steps)

    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    n_states = int(env.unwrapped.observation_space.n)
    obs_adapter = _obs_adapter_factory(n_states)

    print(f"[Env] NoisyGridworld train obs_space={env.observation_space}, eval obs_space={eval_env.observation_space}")
    print(f"[Env] NoisyGridworld action_space={env.action_space}")

    return env, eval_env, obs_adapter
