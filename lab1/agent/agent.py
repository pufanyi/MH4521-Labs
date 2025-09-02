from abc import ABC, abstractmethod
from lab1.bandit.base import Bandit
from tqdm import trange
from pydantic import BaseModel

class Results(BaseModel):
    selected_arm: int
    reward: float

class Agent(ABC):
    @abstractmethod
    def __init__(self, bandit: Bandit):
        self.bandit = bandit

    @abstractmethod
    def play(self) -> Results:
        raise NotImplementedError
    
    def evaluate(self, num_rounds: int):
        regret = 0.0
        total_reward = 0.0
        for i in trange(num_rounds):
            result = self.play()
            regret += self.bandit.best_arm_mean - self.bandit.arms[result.selected_arm].mean()
            total_reward += result.reward
            print(f"Round {i+1}: Selected arm = {result.selected_arm} Reward = {result.reward} Regret = {regret} Total reward = {total_reward} Best arm = {self.bandit.best_arm}")
        return regret
