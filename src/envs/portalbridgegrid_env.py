from typing import Tuple, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from .frozenlake_env import _obs_adapter_factory


class PortalBridgeGrid(gym.Env):
    """
    Gridworld with a long safe detour and a short risky portal edge.

    - Deterministic moves with walls.
    - A vertical barrier wall splits the map (left/right) with a single opening at the top.
    - A single risky "portal edge" exists at (portal_from, RIGHT):
        succeeds -> teleports to portal_to
        fails    -> terminal hole with hole_reward

    This isolates one stochastic (s,a) with two outcomes (siblings).
    """
    metadata = {"render_modes": ["ansi", "human"]}

    # Actions: 0=LEFT,1=DOWN,2=RIGHT,3=UP
    def __init__(
        self,
        nrow: int = 6,
        ncol: int = 9,
        p_fail: float = 0.002,
        step_cost: float = -0.1,
        goal_reward: float = 20.0,
        hole_reward: float = -100.0,
        render_mode=None,
    ):
        super().__init__()
        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.p_fail = float(p_fail)
        self.step_cost = float(step_cost)
        self.goal_reward = float(goal_reward)
        self.hole_reward = float(hole_reward)
        self.render_mode = render_mode

        self.observation_space = spaces.Discrete(self.nrow * self.ncol)
        self.action_space = spaces.Discrete(4)

        # Key locations (defaults tuned for nrow=6,ncol=9)
        self.start = (3, 0)
        self.goal = (3, self.ncol - 1)

        # Hole is not normally reachable; only via portal failure
        self.hole = (self.nrow - 1, self.ncol // 2)

        # Barrier wall at column barrier_c except opening at opening_r
        self.barrier_c = self.ncol // 2  # e.g. 4 when ncol=9
        self.opening_r = 1               # top opening row

        # Portal edge: from left-of-barrier on the mid row to right-of-barrier
        self.portal_from = (3, self.barrier_c - 1)     # e.g. (3,3)
        self.portal_to   = (3, self.barrier_c + 1)     # e.g. (3,5)
        self.portal_action = 2  # RIGHT

        # Build walls
        self.walls = set()
        # Barrier column: block rows 1..(nrow-2), except opening
        for r in range(0, self.nrow):
            if r == self.opening_r:
                continue
            self.walls.add((r, self.barrier_c))

        self.walls.add(self.hole)

        self.state = self._to_s(self.start)
        self.np_random, _ = seeding.np_random(None)

        self.took_portal = False

    def _to_s(self, rc):
        r, c = rc
        return int(r * self.ncol + c)

    def _to_rc(self, s):
        r, c = divmod(int(s), self.ncol)
        return int(r), int(c)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self.state = self._to_s(self.start)
        self.took_portal = False
        return int(self.state), {}


    def step(self, action):
        assert self.action_space.contains(action)
        a = int(action)
        r, c = self._to_rc(self.state)

        # Terminal guard
        if (r, c) == self.goal or (r, c) == self.hole:
            info = {"took_portal": bool(getattr(self, "took_portal", False))}
            return int(self.state), 0.0, True, False, info

        terminated = False
        truncated = False

        # Default reward is per-step cost
        reward = float(self.step_cost)

        # Risky portal edge override
        if (r, c) == self.portal_from and a == self.portal_action:
            # Mark portal usage for this episode (regardless of outcome)
            self.took_portal = True

            if self.np_random.random() < self.p_fail:
                ns = self._to_s(self.hole)
                self.state = ns
                info = {"took_portal": True}
                return int(ns), float(self.hole_reward), True, False, info
            else:
                ns = self._to_s(self.portal_to)
                self.state = ns
                info = {"took_portal": True}
                return int(ns), reward, False, False, info

        # Deterministic move
        drdc = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[a]
        nr, nc = r + drdc[0], c + drdc[1]

        # Bounds
        if not (0 <= nr < self.nrow and 0 <= nc < self.ncol):
            nr, nc = r, c

        # Walls
        if (nr, nc) in self.walls:
            nr, nc = r, c

        # Goal check
        if (nr, nc) == self.goal:
            self.state = self._to_s((nr, nc))
            info = {"took_portal": bool(self.took_portal)}
            return int(self.state), float(self.goal_reward), True, False, info

        # Non-terminal transition
        self.state = self._to_s((nr, nc))
        info = {"took_portal": bool(getattr(self, "took_portal", False))}
        return int(self.state), reward, terminated, truncated, info


    def render(self):
        if self.render_mode not in ("ansi", "human"):
            return None

        grid = [["." for _ in range(self.ncol)] for _ in range(self.nrow)]
        for (wr, wc) in self.walls:
            grid[wr][wc] = "#"

        sr, sc = self.start
        gr, gc = self.goal
        hr, hc = self.hole
        pr, pc = self.portal_from
        tr, tc = self.portal_to

        grid[sr][sc] = "S"
        grid[gr][gc] = "G"
        grid[hr][hc] = "H"
        grid[pr][pc] = "P"
        grid[tr][tc] = "p"

        ar, ac = self._to_rc(self.state)
        grid[ar][ac] = "A"  # agent

        s = "\n".join("".join(row) for row in grid)
        if self.render_mode == "human":
            print(s)
        return s


def make_portalbridgegrid(cfg, seed: int) -> Tuple[gym.Env, gym.Env, Callable]:
    env = PortalBridgeGrid(
        nrow=int(getattr(cfg, "nrow", 6)),
        ncol=int(getattr(cfg, "ncol", 9)),
        p_fail=float(getattr(cfg, "p_fail", 0.002)),
        step_cost=float(getattr(cfg, "step_cost", -0.1)),
        goal_reward=float(getattr(cfg, "goal_reward", 20.0)),
        hole_reward=float(getattr(cfg, "hole_reward", -100.0)),
        render_mode=getattr(cfg, "render_mode", None),
    )
    eval_env = PortalBridgeGrid(
        nrow=int(getattr(cfg, "nrow", 6)),
        ncol=int(getattr(cfg, "ncol", 9)),
        p_fail=float(getattr(cfg, "p_fail", 0.002)),
        step_cost=float(getattr(cfg, "step_cost", -0.1)),
        goal_reward=float(getattr(cfg, "goal_reward", 20.0)),
        hole_reward=float(getattr(cfg, "hole_reward", -100.0)),
        render_mode=getattr(cfg, "render_mode", None),
    )

    env = RecordEpisodeStatistics(env)
    eval_env = RecordEpisodeStatistics(eval_env)

    max_steps = int(getattr(cfg, "max_episode_steps", 200))
    env = TimeLimit(env, max_episode_steps=max_steps)
    eval_env = TimeLimit(eval_env, max_episode_steps=max_steps)

    env.reset(seed=seed)
    eval_env.reset(seed=seed + 123)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 123)

    n_states = int(env.unwrapped.observation_space.n)
    obs_adapter = _obs_adapter_factory(n_states)
    return env, eval_env, obs_adapter
