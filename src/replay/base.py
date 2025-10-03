from abc import ABC, abstractmethod
from typing import Dict


class ReplayBuffer(ABC):
    @abstractmethod
    def add(self, obs, action, reward, next_obs, terminated, truncated):
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> Dict:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    def update_priorities(self, indices, priorities):
        # For PER
        return
