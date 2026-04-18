from collections import defaultdict
from typing import Dict, Tuple, Optional

import numpy as np


class TabularDynamicsModel:
    """
    Exact empirical joint model over outcome tuples z = (s_next, reward, terminated, truncated)
    for each discrete (s, a).

    This matches the intended MODEL semantics for the paper: when (s, a) is selected,
    sample a full stored outcome from the empirical joint distribution induced by the buffer.
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.outcome_counts: Dict[Tuple[int, int], Dict[Tuple[int, float, bool, bool], int]] = defaultdict(lambda: defaultdict(int))

    def observe(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        terminated: bool,
        truncated: bool,
    ):
        key = (int(s), int(a))
        z = (int(s_next), float(r), bool(terminated), bool(truncated))
        self.outcome_counts[key][z] += 1

    def sample(self, s: int, a: int, default: Optional[Tuple[int, float, bool, bool]] = None):
        """
        Sample a full empirical outcome tuple for the queried (s, a).

        If (s, a) was never observed, return `default` if provided, otherwise a neutral fallback.
        """
        key = (int(s), int(a))
        counts_dict = self.outcome_counts.get(key, None)

        if not counts_dict:
            if default is not None:
                return default
            return int(s), 0.0, False, False

        outcomes = list(counts_dict.keys())
        counts = np.asarray(list(counts_dict.values()), dtype=np.float64)
        probs = counts / counts.sum()
        idx = int(np.random.choice(len(outcomes), p=probs))
        s_next, r, terminated, truncated = outcomes[idx]
        return int(s_next), float(r), bool(terminated), bool(truncated)
