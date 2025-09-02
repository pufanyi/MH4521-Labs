from abc import ABC, abstractmethod
from lab1.arm.base import Arm
import numpy as np

class Bandit(ABC):
    def __post_init__(self, n_arms: int):
        self.n_arms = n_arms
        self.arms = [self.generate_arm() for _ in range(n_arms)]
        self.best_arm = np.argmax([arm.mean() for arm in self.arms])
        self.best_arm_mean = self.arms[self.best_arm].mean()

    @abstractmethod
    def generate_arm(self) -> Arm:
        raise NotImplementedError

    def pull(self, arm: int) -> float:
        return self.arms[arm].pull()
