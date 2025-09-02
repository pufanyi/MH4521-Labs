import numpy as np

class GaussianArm():
    def __init__(self, mean: float = 0, std: float = 1):
        self.mean = mean
        self.std = std

    def pull(self):
        return np.random.normal(self.mean, self.std)
