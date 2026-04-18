from collections import Counter
import numpy as np


class TabularDynamicsModel:
    """
    Empirical outcome model over full outcomes z = (s_next, r, terminated, truncated)
    for each exact group (s, a).

    This matches the paper's MODEL definition much better than storing only:
      - counts over s'
      - mean reward per s'
      - marginal terminal probabilities per s'
    """

    def __init__(self, n_states: int, n_actions: int, reward_decimals: int | None = 8):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.reward_decimals = reward_decimals

        # outcome_counts[s][a] is a Counter over tuples:
        #   (s_next, reward, terminated, truncated) -> count
        self.outcome_counts = [
            [Counter() for _ in range(self.n_actions)]
            for _ in range(self.n_states)
        ]

    def _canon_reward(self, r: float) -> float:
        r = float(r)
        if self.reward_decimals is not None:
            r = round(r, int(self.reward_decimals))
        return r

    def observe(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        terminated: bool,
        truncated: bool,
    ):
        s = int(s)
        a = int(a)
        s_next = int(s_next)

        outcome = (
            s_next,
            self._canon_reward(float(r)),
            bool(terminated),
            bool(truncated),
        )
        self.outcome_counts[s][a][outcome] += 1

    def sample(self, s: int, a: int, default=None):
        """
        Sample a synthetic outcome from the empirical frequency model P_hat(. | s, a).

        If unseen, fall back to `default` if provided, else a neutral self-loop.
        """
        s = int(s)
        a = int(a)

        ctr = self.outcome_counts[s][a]
        if not ctr:
            if default is not None:
                return default
            return s, 0.0, False, False

        outcomes = list(ctr.keys())
        freqs = np.asarray(list(ctr.values()), dtype=np.float64)
        probs = freqs / freqs.sum()

        k = int(np.random.choice(len(outcomes), p=probs))
        s_next, r, terminated, truncated = outcomes[k]
        return int(s_next), float(r), bool(terminated), bool(truncated)