from typing import Tuple, Callable, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from gymnasium.utils import seeding

from .frozenlake_env import _obs_adapter_factory


class ShortcutMazeEnv(gym.Env):
    """
    Stochastic Shortcuts Gridworld (Bridge Version) to stress-test PER self-selection bias.

    Two routes to the goal:
      (1) Safe path: a long, deterministic, single-corridor "snake" maze to the goal (reliable +goal_reward).
      (2) Risky shortcut: a short corridor to a fragile bridge decision cell B.
          At B, action RIGHT triggers a stochastic bridge crossing:
            success w.p. risky_success_prob -> jump to far side near goal + risky_success_reward
            fail otherwise -> episode terminates + risky_fail_penalty

    Layout symbols:
      S : start
      G : goal
      # : wall
      . : normal traversable cell
      B : bridge decision cell (special transition only when taking RIGHT while standing on B)

    Actions:
      0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP   (FrozenLake order)

    Reward:
      - step_penalty each step (default 0.0)
      - +goal_reward upon entering G (terminated=True)
      - +risky_success_reward on bridge success (episode continues)
      - +risky_fail_penalty on bridge failure (terminated=True)

    Notes:
      - Deterministic dynamics everywhere except the bridge transition at (bridge_pos, RIGHT).
      - Designed so the safe route is optimal in expectation (with typical params),
        but PER may over-focus on rare bridge successes due to replay sampling feedback.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        size: int = 32,
        goal_reward: float = 10.0,
        step_penalty: float = 0.0,
        risky_success_prob: float = 0.02,
        risky_success_reward: float = 450.0,
        risky_fail_penalty: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if size < 9:
            raise ValueError("size must be >= 9 to create a meaningful snake + shortcut + bridge.")

        self.size = int(size)
        self.height = self.size
        self.width = self.size

        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.risky_success_prob = float(risky_success_prob)
        self.risky_success_reward = float(risky_success_reward)
        self.risky_fail_penalty = float(risky_fail_penalty)
        self.render_mode = render_mode

        self._layout = self._generate_layout(self.size)
        assert len(self._layout) == self.height
        assert all(len(row) == self.width for row in self._layout), "Non-rectangular layout"

        # Locate special cells
        self.start_pos = self._find_symbol("S")
        self.goal_pos = self._find_symbol("G")
        self.bridge_pos = self._find_symbol("B")

        # Bridge landing (far side near goal).
        # By construction we open a short bottom corridor to the goal, so landing near bottom-right is fine.
        # This is the "other side of the lake" in the report narrative.
        self.bridge_success_pos = (self.height - 1, self.width - 2)  # one step left of goal on bottom row

        # Spaces
        n_states = self.height * self.width
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(4)  # LEFT, DOWN, RIGHT, UP

        self.np_random, _ = seeding.np_random(None)
        self.agent_pos = self.start_pos

    def _generate_layout(self, size: int) -> List[str]:
        """
        Construct a maze with:
        - Safe path: long snake corridor (DISJOINT from risky after the start).
        - Risky path: short corridor down the left edge to bridge cell B at column 0.
        - Bridge crossing occurs ONLY via special transition at (B, RIGHT).

        Key constraint enforced here:
        After the first move from S, the safe and risky branches do not connect.
        """
        grid = np.full((size, size), "#", dtype="<U1")

        # Place start/goal
        grid[0, 0] = "S"
        grid[size - 1, size - 1] = "G"

        # Column 1 is a hard wall barrier for all rows except row 0.
        # This prevents the risky branch (col 0) from leaking into the safe snake (cols >= 2).
        for r in range(1, size):
            grid[r, 1] = "#"
        # Allow safe branch entry from S via (0,1) -> (0,2)
        grid[0, 1] = "."

        # Safe snake occupies columns 2..size-2 (never touches column 0/1 after row 0).
        left_c, right_c = 2, size - 2
        if right_c <= left_c:
            raise ValueError("size too small to build safe snake with separator (need size >= 5).")

        # Start by ending at left_c so we can go down near the entry (more intuitive).
        sweep_right = False  # [CHANGED] was True in your earlier version
        for r in range(0, size - 1, 2):  # even rows: carve horizontal segment
            if sweep_right:
                for c in range(left_c, right_c + 1):
                    grid[r, c] = "."
                end_c = right_c
            else:
                for c in range(right_c, left_c - 1, -1):
                    grid[r, c] = "."
                end_c = left_c

            # Connector cell down to next row (single connector)
            if r + 1 < size:
                grid[r + 1, end_c] = "."

            sweep_right = not sweep_right

        # Ensure safe corridor is reachable from S
        grid[0, 2] = "."

        # Bottom row: open from col=2 to goal (keep col 0/1 closed to preserve disjointness)
        for c in range(2, size):
            if grid[size - 1, c] == "#":
                grid[size - 1, c] = "."
        grid[size - 1, size - 1] = "G"

        # Ensure bridge success landing is traversable (one left of goal)
        grid[size - 1, size - 2] = "."

        # Risky path is a short vertical corridor down col 0 to B.
        # Place B on col 0 so there is no need to carve through the separator.
        bridge_r = max(2, size // 4)      # short risky route
        bridge_r = min(bridge_r, size - 3)

        for r in range(0, bridge_r + 1):
            if grid[r, 0] == "#":
                grid[r, 0] = "."
        grid[bridge_r, 0] = "B"

        return ["".join(grid[r, :].tolist()) for r in range(size)]


    # -----------------------
    # Helpers
    # -----------------------
    def _find_symbol(self, sym: str) -> Tuple[int, int]:
        for r, row in enumerate(self._layout):
            for c, ch in enumerate(row):
                if ch == sym:
                    return (r, c)
        raise ValueError(f"Symbol {sym!r} not found in layout")

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * self.width + c

    def _cell(self, pos: Tuple[int, int]) -> str:
        r, c = pos
        return self._layout[r][c]

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        return self._cell(pos) == "#"

    # -----------------------
    # Gym API
    # -----------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self.agent_pos = self.start_pos
        obs = self._pos_to_state(self.agent_pos)
        info = {}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward = self.step_penalty
        terminated = False
        truncated = False
        info = {"events": {
            "goal_reached": 0,
            "bridge_visit": 0,
            "bridge_take_right": 0,
            "bridge_success": 0,
            "bridge_fail": 0,
        }}

        if self.agent_pos == self.bridge_pos:
            info["events"]["bridge_visit"] = 1

        # Special bridge transition: only triggers when standing on B and taking RIGHT (action=2)
        if self.agent_pos == self.bridge_pos and action == 2:
            info["events"]["bridge_take_right"] = 1
            if float(self.np_random.random()) < self.risky_success_prob:
                # Successful crossing: land on far side near goal and get jackpot
                self.agent_pos = self.bridge_success_pos
                reward += self.risky_success_reward
                info["events"]["bridge_success"] = 1
                info["bridge"] = "success"
            else:
                # Bridge collapses: terminal failure
                reward += self.risky_fail_penalty
                terminated = True
                info["events"]["bridge_fail"] = 1
                info["bridge"] = "fail"

            obs = self._pos_to_state(self.agent_pos)
            return obs, float(reward), terminated, truncated, info

        # Otherwise, deterministic grid movement (FrozenLake action order)
        r, c = self.agent_pos
        if action == 0:          # LEFT
            new_r, new_c = r, c - 1
        elif action == 1:        # DOWN
            new_r, new_c = r + 1, c
        elif action == 2:        # RIGHT
            new_r, new_c = r, c + 1
        else:                   # UP
            new_r, new_c = r - 1, c

        # Bounds
        if not (0 <= new_r < self.height and 0 <= new_c < self.width):
            new_r, new_c = r, c

        new_pos = (new_r, new_c)

        # Walls
        if self._is_wall(new_pos):
            new_pos = (r, c)

        cell = self._cell(new_pos)

        # Goal
        if cell == "G":
            reward += self.goal_reward
            terminated = True

            info["events"]["goal_reached"] = 1

        self.agent_pos = new_pos
        obs = self._pos_to_state(self.agent_pos)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        lines: List[str] = []
        for r in range(self.height):
            row_chars: List[str] = []
            for c in range(self.width):
                pos = (r, c)
                ch = self._layout[r][c]
                row_chars.append("A" if pos == self.agent_pos else ch)
            lines.append("".join(row_chars))
        out = "\n".join(lines)
        if self.render_mode == "human":
            print(out)
        return out

    def close(self):
        pass


def make_shortcut_maze(cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    """
    Factory mirroring your make_noisy_gridworld() integration.

    Expected cfg fields (with defaults):
      - size: int (default 32)
      - goal_reward: float (default 10.0)
      - step_penalty: float (default 0.0)
      - risky_success_prob: float (default 0.02)
      - risky_success_reward: float (default 450.0)
      - risky_fail_penalty: float (default 0.0)
      - max_episode_steps: int (default 2000)
      - render_mode: Optional[str]
    """
    kwargs = dict(
        size=int(getattr(cfg, "size", 32)),
        goal_reward=float(getattr(cfg, "goal_reward", 10.0)),
        step_penalty=float(getattr(cfg, "step_penalty", 0.0)),
        risky_success_prob=float(getattr(cfg, "risky_success_prob", 0.02)),
        risky_success_reward=float(getattr(cfg, "risky_success_reward", 450.0)),
        risky_fail_penalty=float(getattr(cfg, "risky_fail_penalty", 0.0)),
        render_mode=getattr(cfg, "render_mode", None),
    )

    env = ShortcutMazeEnv(**kwargs)
    eval_env = ShortcutMazeEnv(**kwargs)

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    max_steps = int(getattr(cfg, "max_episode_steps", 2000))
    env = TimeLimit(env, max_episode_steps=max_steps)
    eval_env = TimeLimit(eval_env, max_episode_steps=max_steps)

    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    n_states = int(env.unwrapped.observation_space.n)
    obs_adapter = _obs_adapter_factory(n_states)

    print(f"[Env] ShortcutMaze (Bridge) train obs_space={env.observation_space}, eval obs_space={eval_env.observation_space}")
    print(f"[Env] ShortcutMaze (Bridge) action_space={env.action_space}")
    print(f"[Env] ShortcutMaze (Bridge) size={kwargs['size']} bridge_p={kwargs['risky_success_prob']} "
          f"jackpot={kwargs['risky_success_reward']} goal={kwargs['goal_reward']} fail_pen={kwargs['risky_fail_penalty']}")

    return env, eval_env, obs_adapter
