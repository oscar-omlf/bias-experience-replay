from abc import ABC, abstractmethod
from typing import Dict, Sequence, List


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
        pass

    def fetch(self, indices: Sequence[int]) -> Dict:
        pass

    def sibling_groups(
            self,
            indices: Sequence[int],
            include_self: bool,
            min_group: int,
            max_group: int
        ) -> List[List[int]]:
        return [[] for _ in indices]

