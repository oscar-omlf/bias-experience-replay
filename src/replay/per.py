from typing import Dict, Sequence, List
import numpy as np
from collections import defaultdict
from .base import ReplayBuffer

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def add(self, priority: float):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] += change
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_priority(self) -> float:
        return self.tree[0]

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, obs_shape, alpha: float = 0.6, beta: float = 0.4, beta_anneal_steps: int = 0, device: str = "cpu"):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta_start = beta
        self.beta = beta
        self.beta_anneal_steps = beta_anneal_steps if beta_anneal_steps > 0 else None
        self.device = device
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.pos = 0
        self.full = False
        self.current_size = 0
        self.step = 0

        if isinstance(obs_shape, int):
            self.obs = np.zeros((capacity,), dtype=np.int64)
        else:
            self.obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)

        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.float32)
        self.truncated = np.zeros((capacity,), dtype=np.float32)

        self.by_sa = defaultdict(list)
        self.idx_to_key = [None] * capacity

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        old_key = self.idx_to_key[self.pos]
        if old_key is not None:
            lst = self.by_sa.get(old_key, None)
            if lst is not None:
                try:
                    lst.remove(self.pos)
                except ValueError:
                    pass

        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.terminated[self.pos] = float(terminated)
        self.truncated[self.pos] = float(truncated)

        if self.obs.ndim == 1:
            s_key = int(self.obs[self.pos])
        else:
            s_key = self.obs[self.pos].tobytes()

        # s_key = self.idx_to_key[self.obs[self.pos]]
        key = (s_key, int(self.actions[self.pos]))

        self.by_sa[key].append(self.pos)
        self.idx_to_key[self.pos] = key
        
        priority = max(self.max_priority, 1e-6) ** self.alpha
        self.tree.add(priority)
        
        self.pos = (self.pos + 1) % self.capacity
        if not self.full:
            self.current_size += 1
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int) -> Dict:
        indices = []
        priorities = []
        total_prio = self.tree.total_priority()
        segment = total_prio / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            tree_idx, p, data_idx = self.tree.get_leaf(v)
            priorities.append(p)
            indices.append(data_idx)
        probs = np.array(priorities) / total_prio
        n = self.current_size
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize
        self.step += 1
        if self.beta_anneal_steps:
            self.beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.step / self.beta_anneal_steps)
        batch = dict(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
            indices=np.array(indices),
            weights=weights,
        )
        return batch
    
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
    
    def sibling_groups(
        self,
        indices: Sequence[int],
        include_self: bool,
        min_group: int,
        max_group: int
    ) -> List[List[int]]:
        groups = []
        for idx in indices:
            key = self.idx_to_key[idx]
            if key is None:
                groups.append([])
                continue
            lst = self.by_sa.get(key, [])
            if include_self:
                g = list(lst)
            else:
                g = [j for j in lst if j != idx]
            if max_group and len(g) > max_group:
                g = np.random.choice(g, size=max_group, replace=False).tolist()
            if len(g) < min_group:
                g = []  # caller will fallback to original idx
            groups.append(g)
        return groups

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.max_priority = max(self.max_priority, prio)
            prio = max(prio, 1e-6) ** self.alpha
            self.tree.update(idx + self.capacity - 1, prio)

    def __len__(self) -> int:
        return self.current_size
