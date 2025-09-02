import numpy as np

from lab1.agent.agent import Agent, Results
from lab1.bandit.base import Bandit


class UCBAgent(Agent):
    def __init__(
        self,
        bandit: Bandit,
        delta: float,
        c: float,
        eps: float = 0,
        log_to_wandb: bool = True,
    ):
        super().__init__(bandit, log_to_wandb)
        self.delta = delta
        self.c = c
        self.eps = eps
        if self.eps < 0 or self.eps > 1:
            raise ValueError("eps must be between 0 and 1")
        if self.c <= 0:
            raise ValueError("c must be greater than 0")
        self.ucb_values = [float("inf")] * self.bandit.n_arms
        self.num_pulls = [0] * self.bandit.n_arms
        self.q_values = [0] * self.bandit.n_arms

    def play(self) -> Results:
        if np.random.random() < self.eps:
            selected_arm = np.random.randint(0, self.bandit.n_arms)
        else:
            selected_arm = np.argmax(self.ucb_values)
        reward = self.bandit.pull(selected_arm)
        self.num_pulls[selected_arm] += 1
        self.q_values[selected_arm] = (
            self.q_values[selected_arm]
            + (reward - self.q_values[selected_arm]) / self.num_pulls[selected_arm]
        )
        self.ucb_values[selected_arm] = self.q_values[selected_arm] + self.c * np.sqrt(
            2 * np.log(1.0 / self.delta) / self.num_pulls[selected_arm]
        )
        return Results(selected_arm=selected_arm, reward=reward)
