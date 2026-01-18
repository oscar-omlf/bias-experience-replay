import random
from typing import Dict, Sequence, List, Optional
from collections import defaultdict
import numpy as np
from .base import ReplayBuffer


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    def total_priority(self) -> float:
        return float(self.tree[0])

    def leaf_idx(self, data_idx: int) -> int:
        return int(data_idx) + self.capacity - 1

    def update(self, data_idx: int, priority: float):
        """
        Set leaf(data_idx) = priority and propagate delta up the tree.
        """
        li = self.leaf_idx(data_idx)
        change = priority - self.tree[li]
        self.tree[li] = priority
        while li != 0:
            li = (li - 1) // 2
            self.tree[li] += change

    def get_leaf(self, v: float):
        """
        Find smallest leaf index such that prefix sum >= v.
        """
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left]:
                parent_idx = left
            else:
                v -= self.tree[left]
                parent_idx = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), int(data_idx)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        obs_shape,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_anneal_steps: int = 0,
        device: str = "cpu",
        keyer=None,  # NEW
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)

        self.beta_start = float(beta)
        self.beta = float(beta)
        self.beta_anneal_steps = int(beta_anneal_steps) if beta_anneal_steps and beta_anneal_steps > 0 else None

        self.device = device
        self.tree = SumTree(self.capacity)

        self.max_priority = 1.0

        self.pos = 0
        self.full = False
        self.current_size = 0
        self.step = 0

        if isinstance(obs_shape, int):
            self.obs = np.zeros((self.capacity,), dtype=np.int64)
        else:
            self.obs = np.zeros((self.capacity,) + tuple(obs_shape), dtype=np.float32)

        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.terminated = np.zeros((self.capacity,), dtype=np.float32)
        self.truncated = np.zeros((self.capacity,), dtype=np.float32)

        # Sibling tracking
        self.by_sa = defaultdict(list)      # key -> list[int indices]
        self.idx_to_key = [None] * self.capacity
        self.idx_to_pos = np.full((self.capacity,), -1, dtype=np.int64)

        # Group priority mass: s_g = sum_{i in group} p_i^alpha (leaf values)
        self.group_mass = defaultdict(float)

        # group keyer (e.g., VQ-VAE)
        self.keyer = keyer

        # Debug
        self.debug_key = None

    def rebuild_groups(self):
        """
        Recompute by_sa, idx_to_key, idx_to_pos, and group_mass from scratch.

        WARNING: This changes group assignments for existing transitions if self.keyer changed.
        Prefer freezing the keyer for theory-faithful runs.
        """
        self.by_sa.clear()
        self.group_mass.clear()
        self.idx_to_key = [None] * self.capacity
        self.idx_to_pos[:] = -1

        n = self._valid_n()
        for idx in range(n):
            key = self._make_key(self.obs[idx], int(self.actions[idx]))
            self.idx_to_key[idx] = key
            lst = self.by_sa[key]
            lst.append(idx)
            self.idx_to_pos[idx] = len(lst) - 1
            leaf = float(self.tree.tree[self.tree.leaf_idx(idx)])
            self.group_mass[key] += leaf

    def _make_key(self, obs, action: int):
        """
        Key for sibling grouping.

        If self.keyer is provided, use group_id = keyer(obs).
        Otherwise:
        - if scalar-discrete buffer: group_id = int(obs)
        - else: group_id = obs.tobytes() (exact-match only)
        """
        if self.keyer is not None:
            gid = int(self.keyer(obs))
            return (gid, int(action))

        if self.obs.ndim == 1:
            s_key = int(obs)
        else:
            arr = np.asarray(obs, dtype=self.obs.dtype)
            s_key = arr.tobytes()
        return (s_key, int(action))

    def set_debug_key(self, obs, action):
        self.debug_key = self._make_key(obs, int(action))

    def debug_snapshot(self):
        if self.debug_key is None:
            return None
        lst = self.by_sa.get(self.debug_key, [])
        if not lst:
            return {"debug_group_size": 0}
        # leaf values are already p^alpha
        prios = [float(self.tree.tree[self.tree.leaf_idx(i)]) for i in lst]
        return {
            "debug_group_size": len(lst),
            "debug_indices": list(lst),
            "debug_priorities": prios,
            "debug_group_mass": float(self.group_mass.get(self.debug_key, 0.0)),
        }

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        # If overwriting, remove old membership + subtract old leaf mass
        old_key = self.idx_to_key[self.pos]
        old_leaf = float(self.tree.tree[self.tree.leaf_idx(self.pos)])

        if old_key is not None:
            lst = self.by_sa.get(old_key, None)
            if lst is not None:
                p = int(self.idx_to_pos[self.pos])
                # O(1) removal if position bookkeeping is consistent
                if 0 <= p < len(lst) and int(lst[p]) == int(self.pos):
                    last = int(lst[-1])
                    lst[p] = last
                    self.idx_to_pos[last] = p
                    lst.pop()
                    self.idx_to_pos[self.pos] = -1
                else:
                    # Fallback to linear search if something went wrong
                    try:
                        lst.remove(self.pos)
                    except ValueError:
                        pass
                    self.idx_to_pos[self.pos] = -1

                if len(lst) == 0:
                    self.by_sa.pop(old_key, None)

            self.group_mass[old_key] -= old_leaf
            if self.group_mass[old_key] <= 0.0:
                self.group_mass.pop(old_key, None)

        # Write transition
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.terminated[self.pos] = float(terminated)
        self.truncated[self.pos] = float(truncated)

        key = self._make_key(self.obs[self.pos], int(self.actions[self.pos]))
        lst = self.by_sa[key]
        lst.append(self.pos)
        self.idx_to_pos[self.pos] = len(lst) - 1
        self.idx_to_key[self.pos] = key

        # New leaf priority (stored as p^alpha)
        new_leaf = float(max(self.max_priority, 1e-6) ** self.alpha)
        self.tree.update(self.pos, new_leaf)
        self.group_mass[key] += new_leaf

        # Advance
        self.pos = (self.pos + 1) % self.capacity
        if not self.full:
            self.current_size += 1
            if self.current_size == self.capacity:
                self.full = True

    def __len__(self) -> int:
        return int(self.current_size)

    def _valid_n(self) -> int:
        return int(self.current_size)

    def sample(self, batch_size: int) -> Dict:
        n = self._valid_n()
        if n <= 0:
            raise RuntimeError("PER sample() called with empty buffer.")

        total = self.tree.total_priority()
        if total <= 0.0:
            # Fallback: uniform
            idxs = np.random.randint(0, n, size=batch_size)
            weights = np.ones((batch_size,), dtype=np.float32)
            return dict(
                obs=self.obs[idxs],
                actions=self.actions[idxs],
                rewards=self.rewards[idxs],
                next_obs=self.next_obs[idxs],
                terminated=self.terminated[idxs],
                truncated=self.truncated[idxs],
                indices=idxs.astype(np.int64),
                weights=weights,
                sampling_probs=np.full((batch_size,), 1.0 / n, dtype=np.float32),
            )

        indices = np.empty((batch_size,), dtype=np.int64)
        priorities = np.empty((batch_size,), dtype=np.float64)

        segment = total / float(batch_size)
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            _, p, data_idx = self.tree.get_leaf(v)

            # Safety: clamp to valid prefix of buffer when not full
            if (not self.full) and (data_idx >= n):
                data_idx = int(np.random.randint(0, n))
                p = float(self.tree.tree[self.tree.leaf_idx(data_idx)])

            indices[i] = data_idx
            priorities[i] = p

        probs = (priorities / total).astype(np.float32)

        # Standard PER IS weights: (n * P(i))^{-beta}
        weights = (n * probs) ** (-self.beta)
        wmax = float(weights.max()) if weights.size > 0 else 1.0
        if wmax > 0:
            weights /= wmax

        # Anneal beta
        self.step += 1
        if self.beta_anneal_steps:
            frac = min(1.0, self.step / float(self.beta_anneal_steps))
            self.beta = self.beta_start + (1.0 - self.beta_start) * frac

        return dict(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
            indices=indices,
            weights=weights.astype(np.float32),
            sampling_probs=probs,
        )

    def fetch(self, indices: Sequence[int]) -> Dict:
        idx = np.asarray(indices, dtype=np.int64)
        return dict(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
            indices=idx,
        )

    def update_priorities(self, indices, priorities):
        # priorities are RAW (abs TD + eps) before exponent alpha
        for idx, prio_raw in zip(indices, priorities):
            idx = int(idx)
            if idx < 0 or idx >= self.capacity:
                continue
            if (not self.full) and idx >= self.current_size:
                continue

            key = self.idx_to_key[idx]
            if key is None:
                continue

            # Update max raw priority
            prio_raw = float(prio_raw)
            self.max_priority = max(self.max_priority, prio_raw)

            new_leaf = float(max(prio_raw, 1e-6) ** self.alpha)
            li = self.tree.leaf_idx(idx)
            old_leaf = float(self.tree.tree[li])
            delta = new_leaf - old_leaf

            if delta != 0.0:
                self.tree.update(idx, new_leaf)
                self.group_mass[key] += delta
                if self.group_mass[key] <= 0.0:
                    self.group_mass.pop(key, None)

    def sibling_groups(
        self,
        indices: Sequence[int],
        include_self: bool,
        min_group: int,
        max_group: int
    ) -> List[List[int]]:
        groups: List[List[int]] = []
        key_to_lst = {}

        cap = int(max_group) if max_group and max_group > 0 else None
        for idx in indices:
            idx = int(idx)
            key = self.idx_to_key[idx]
            if key is None:
                groups.append([])
                continue

            if key not in key_to_lst:
                key_to_lst[key] = self.by_sa.get(key, [])

            lst = key_to_lst[key]
            n = len(lst)
            if n == 0:
                groups.append([])
                continue

            if cap is None:
                # Avoid this in large runs; keep max_group > 0 in config.
                if include_self:
                    g = list(lst)
                else:
                    g = [j for j in lst if j != idx]
            else:
                want = cap
                if include_self:
                    if n <= want:
                        g = list(lst)
                    else:
                        g = random.sample(lst, want)
                else:
                    if n <= want + 1:
                        g = [j for j in lst if j != idx]
                    else:
                        g_set = set()
                        while len(g_set) < want:
                            cand = lst[np.random.randint(0, n)]
                            if cand != idx:
                                g_set.add(int(cand))
                        g = list(g_set)

            if len(g) < int(min_group):
                g = []
            groups.append(g)

        return groups

    def compute_group_is_weights(self, indices: Sequence[int], beta: Optional[float] = None, normalize: bool = True):
        """
        Correct IS weights for SAMPLE/AVG/MODEL where effective sampling is:
          choose group g with prob s_g / S, then uniform within group.

        w_g(beta) = ((n_g * S) / (n * s_g))^beta

        Returns:
          weights: np.ndarray [B]
          n_g: np.ndarray [B] group sizes
          s_g: np.ndarray [B] group masses (p^alpha sums)
          S: float total mass
          n: int buffer size
        """
        idx = np.asarray(indices, dtype=np.int64)
        B = idx.shape[0]
        n = self._valid_n()
        S = self.tree.total_priority()

        if beta is None:
            beta = float(self.beta)

        w = np.ones((B,), dtype=np.float64)
        n_g = np.ones((B,), dtype=np.int64)
        s_g = np.ones((B,), dtype=np.float64)

        if n <= 0 or S <= 0.0:
            return w.astype(np.float32), n_g, s_g.astype(np.float32), float(S), int(n)

        for i, di in enumerate(idx):
            di = int(di)
            key = self.idx_to_key[di]
            if key is None:
                continue
            ng = len(self.by_sa.get(key, []))
            sg = float(self.group_mass.get(key, 0.0))

            # Guard against degenerate bookkeeping
            if ng <= 0 or sg <= 0.0:
                continue

            n_g[i] = ng
            s_g[i] = sg
            # ((n_g/n) / (s_g/S))^beta
            w[i] = ((ng * S) / (n * sg)) ** beta

        if normalize and w.size > 0:
            wmax = float(np.max(w))
            if wmax > 0:
                w /= wmax

        return w.astype(np.float32), n_g, s_g.astype(np.float32), float(S), int(n)

    def leaf_value(self, idx: int) -> float:
        return float(self.tree.tree[self.tree.leaf_idx(int(idx))])

    def within_group_used_ratio(self, sampled_indices, used_indices):
        sampled = np.asarray(sampled_indices, dtype=np.int64)
        used = np.asarray(used_indices, dtype=np.int64)
        B = sampled.shape[0]

        ratio = np.ones((B,), dtype=np.float32)
        cond_per = np.ones((B,), dtype=np.float32)
        cond_unif = np.ones((B,), dtype=np.float32)

        for i in range(B):
            si = int(sampled[i])
            ui = int(used[i])

            key = self.idx_to_key[si]
            if key is None:
                continue

            ng = len(self.by_sa.get(key, []))
            sg = float(self.group_mass.get(key, 0.0))
            if ng <= 0 or sg <= 0.0:
                continue

            qu = self.leaf_value(ui)
            p_per = float(qu / sg)
            p_unif = float(1.0 / ng)

            cond_per[i] = p_per
            cond_unif[i] = p_unif
            ratio[i] = float((p_per / p_unif) if p_unif > 0 else 1.0)

        return ratio, cond_per, cond_unif
