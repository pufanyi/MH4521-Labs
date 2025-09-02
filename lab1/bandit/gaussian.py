from lab1.arm.gaussian import GaussianArm
from lab1.bandit.base import Bandit
from lab1.arm.base import Arm
import numpy as np

class GaussianBandit(Bandit):
    def __init__(self, n_arms: int, mean: float, std: float, arms_std: float):
        self.mean = mean
        self.std = std
        self.arms_std = arms_std
        super().__post_init__(n_arms)

    def generate_arm(self) -> Arm:
        arms_mean = np.random.normal(self.mean, self.arms_std, self.n_arms)
        return GaussianArm(arms_mean, self.std)
