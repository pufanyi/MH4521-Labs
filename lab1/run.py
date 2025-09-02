from lab1.agent.eps_agent import EpsAgent
from lab1.agent.etc_agent import EtcAgent
from lab1.bandit.gaussian import GaussianBandit

if __name__ == "__main__":
    bandit = GaussianBandit(n_arms=4, mean=0, std=1, arms_std=0.1)
    agent = EpsAgent(bandit, eps=0.1, alpha=0.1)
    # agent = EtcAgent(bandit, num_trials=10)
    agent.evaluate(num_rounds=1000)
