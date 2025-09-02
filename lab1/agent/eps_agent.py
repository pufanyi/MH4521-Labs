from lab1.bandit.base import Bandit
import numpy as np
from lab1.agent.agent import Agent, Results

class EpsAgent(Agent):
    def __init__(self, bandit: Bandit, eps: float, alpha: float = 0.1, log_to_wandb: bool = True):
        super().__init__(bandit, log_to_wandb)
        self.eps = eps
        if self.eps < 0 or self.eps > 1:
            raise ValueError("eps must be between 0 and 1")
        self.alpha = alpha
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.num_pulls = [0] * self.bandit.n_arms
        self.q_values = [0] * self.bandit.n_arms
        
    def play(self) -> Results:
        if np.random.random() < self.eps:
            selected_arm = np.random.randint(0, self.bandit.n_arms)
        else:
            selected_arm = np.argmax(self.q_values)
        reward = self.bandit.pull(selected_arm)
        self.num_pulls[selected_arm] += 1
        # self.q_values[selected_arm] = self.q_values[selected_arm] + (reward - self.q_values[selected_arm]) / self.num_pulls[selected_arm]
        self.q_values[selected_arm] = self.q_values[selected_arm] + self.alpha * (reward - self.q_values[selected_arm])
        return Results(selected_arm=selected_arm, reward=reward)
