import wandb
from lab1.agent.ucb_agent import UCBAgent
from lab1.bandit.gaussian import GaussianBandit

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="mh4521-bandit-lab",
        config={
            "bandit_arms": 4,
            "bandit_mean": 0,
            "bandit_std": 1,
            "bandit_arms_std": 0.1,
            "num_rounds": 1000,
            "agent_type": "UCB",
            "ucb_delta": 0.1,
            "ucb_c": 2,
            "ucb_eps": 0.1,
        },
        tags=["single_run", "ucb_agent"],
    )

    # Create bandit and agent
    bandit = GaussianBandit(n_arms=4, mean=0, std=1, arms_std=0.1)

    # Uncomment to try different agents:
    # agent = EpsAgent(bandit, eps=0.1, alpha=0.1)
    # agent = EtcAgent(bandit, num_trials=10)
    agent = UCBAgent(bandit, delta=0.1, c=2, eps=0.1)

    # Log bandit information
    wandb.log(
        {
            "bandit_best_arm_mean": bandit.best_arm_mean,
            "bandit_arm_means": [arm.mean() for arm in bandit.arms],
        }
    )

    # Run evaluation
    results = agent.evaluate(num_rounds=1000, log_frequency=50)

    # Log final results
    wandb.log({"experiment_completed": True, "total_rounds": 1000})

    print(f"Final cumulative regret: {results['regret']:.4f}")
    print(f"Final average regret: {results['regret'] / 1000:.4f}")
    print(f"Final total reward: {results['total_reward']:.4f}")
    print(f"Final average reward: {results['total_reward'] / 1000:.4f}")

    wandb.finish()
