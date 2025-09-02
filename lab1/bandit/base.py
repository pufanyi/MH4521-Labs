from abc import ABC, abstractmethod
from lab1.arm.base import Arm

class Bandit(ABC):
    def __post_init__(self, n_arms: int):
        self.n_arms = n_arms
        self.arms = [self.generate_arm() for _ in range(n_arms)]

    @abstractmethod
    def generate_arm(self) -> Arm:
        raise NotImplementedError
