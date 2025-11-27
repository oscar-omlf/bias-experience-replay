import numpy as np


class TabularDynamicsModel:
    """
    Simple tabular model p(s', r, done, trunc | s, a) learned from experience.

    - Assumes discrete states and actions (e.g. FrozenLake).
    - For each (s, a, s'):
        * counts[s, a, s'] how many times we saw that transition
        * reward_sum[s, a, s'] sum of rewards
        * term_counts[s, a, s'] how many times it was terminal
        * trunc_counts[s, a, s'] how many times it was truncated
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions

        self.counts = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
        self.reward_sum = np.zeros((n_states, n_actions, n_states), dtype=np.float32)
        self.term_counts = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
        self.trunc_counts = np.zeros((n_states, n_actions, n_states), dtype=np.int64)

    def observe(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        terminated: bool,
        truncated: bool,
    ):
        self.counts[s, a, s_next] += 1
        self.reward_sum[s, a, s_next] += float(r)
        self.term_counts[s, a, s_next] += int(bool(terminated))
        self.trunc_counts[s, a, s_next] += int(bool(truncated))

    def sample(self, s: int, a: int, default=None):
        """
        Sample a new (s_next, r, terminated, truncated) for (s, a).

        If we have never seen (s, a), fall back to default if given,
        otherwise stay in place with zero reward and no termination.
        """
        counts_sa = self.counts[s, a]
        total = counts_sa.sum()
        if total == 0:
            if default is not None:
                return default
            # Neutral fallback: stay in place, zero reward, not done
            return s, 0.0, False, False

        probs = counts_sa.astype(np.float32) / float(total)
        s_next = int(np.random.choice(self.n_states, p=probs))

        c = max(int(counts_sa[s_next]), 1)
        r_mean = float(self.reward_sum[s, a, s_next] / c)
        term_prob = float(self.term_counts[s, a, s_next] / c)
        trunc_prob = float(self.trunc_counts[s, a, s_next] / c)

        term = np.random.rand() < term_prob
        trunc = np.random.rand() < trunc_prob

        return s_next, r_mean, term, trunc
