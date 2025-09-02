import numpy as np

from lab1.agent.agent import Agent, Results
from lab1.bandit.base import Bandit


class EtcAgent(Agent):
    def __init__(self, bandit: Bandit, num_trials: int, log_to_wandb: bool = True):
        super().__init__(bandit, log_to_wandb)
        self.num_trials = num_trials * self.bandit.n_arms
        self.num_pulls = [0] * self.bandit.n_arms
        self.q_values = [0] * self.bandit.n_arms
        self.attempts = 0

    def play(self) -> Results:
        if self.attempts < self.num_trials:
            selected_arm = self.attempts % self.bandit.n_arms
        else:
            selected_arm = np.argmax(self.q_values)
        reward = self.bandit.pull(selected_arm)
        self.num_pulls[selected_arm] += 1
        self.q_values[selected_arm] = (
            self.q_values[selected_arm]
            + (reward - self.q_values[selected_arm]) / self.num_pulls[selected_arm]
        )
        self.attempts += 1
        return Results(selected_arm=selected_arm, reward=reward)
