import numpy as np
from lab1.arm.base import Arm

class GaussianArm(Arm):
    def __init__(self, mean: float = 0, std: float = 1):
        self._mean = mean
        self._std = std

    def pull(self):
        return np.random.normal(self._mean, self._std)

    def mean(self):
        return self._mean
    
    def std(self):
        return self._std