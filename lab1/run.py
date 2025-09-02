from lab1.agent.ucb_agent import UCBAgent
from lab1.bandit.gaussian import GaussianBandit

if __name__ == "__main__":
    bandit = GaussianBandit(n_arms=10, mean=0, std=1, arms_std=0.1)
    agent = UCBAgent(bandit, delta=0.1, c=2)
    agent.evaluate(num_rounds=1000)
