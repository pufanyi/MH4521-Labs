from abc import ABC, abstractmethod

import wandb
from pydantic import BaseModel
from tqdm import trange

from lab1.bandit.base import Bandit


class Results(BaseModel):
    selected_arm: int
    reward: float


class Agent(ABC):
    @abstractmethod
    def __init__(self, bandit: Bandit, log_to_wandb: bool = True):
        self.bandit = bandit
        self.log_to_wandb = log_to_wandb
        self.step = 0

    @abstractmethod
    def play(self) -> Results:
        raise NotImplementedError

    def evaluate(self, num_rounds: int, log_frequency: int = 10):
        regret = 0.0
        total_reward = 0.0
        cumulative_regret = []
        cumulative_reward = []

        for i in trange(num_rounds, desc="Evaluating agent"):
            result = self.play()
            instant_regret = (
                self.bandit.best_arm_mean - self.bandit.arms[result.selected_arm].mean()
            )
            regret += instant_regret
            total_reward += result.reward

            cumulative_regret.append(regret)
            cumulative_reward.append(total_reward)

            self.step += 1

            # Log to wandb at specified frequency
            if self.log_to_wandb and (i + 1) % log_frequency == 0:
                wandb.log(
                    {
                        "step": self.step,
                        "round": i + 1,
                        "instant_regret": instant_regret,
                        "cumulative_regret": regret,
                        "instant_reward": result.reward,
                        "cumulative_reward": total_reward,
                        "selected_arm": result.selected_arm,
                        "average_regret": regret / (i + 1),
                        "average_reward": total_reward / (i + 1),
                    }
                )

        # Final log
        if self.log_to_wandb:
            wandb.log(
                {
                    "final_cumulative_regret": regret,
                    "final_cumulative_reward": total_reward,
                    "final_average_regret": regret / num_rounds,
                    "final_average_reward": total_reward / num_rounds,
                }
            )

        return {
            "regret": regret,
            "total_reward": total_reward,
            "cumulative_regret": cumulative_regret,
            "cumulative_reward": cumulative_reward,
        }
