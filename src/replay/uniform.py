from typing import Dict, Optional
import numpy as np
from .base import ReplayBuffer


class UniformReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, obs_shape, device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = device
        self.pos = 0
        self.full = False

        if isinstance(obs_shape, int):
            self.obs = np.zeros((capacity,), dtype=np.int64)
        else:
            self.obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int) -> Dict:
        max_idx = self.capacity if self.full else self.pos
        idxs = np.random.randint(0, max_idx, size=batch_size)
        # Adding indices and weights to implement PER later --- returns must match
        batch = dict(
            obs=self.obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs[idxs],
            dones=self.dones[idxs],
            indices=idxs,
            weights=np.ones((batch_size,), dtype=np.float32),
        )
        return batch

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos
