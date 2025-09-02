"""
Multi-Agent Comparison Script with Wandb Integration
This script runs multiple bandit agents and compares their performance using
Weights & Biases.
"""

import time
from typing import Any

import numpy as np

import wandb
from lab1.agent.eps_agent import EpsAgent
from lab1.agent.etc_agent import EtcAgent
from lab1.agent.ucb_agent import UCBAgent
from lab1.bandit.gaussian import GaussianBandit


def run_agent_experiment(
    agent_class,
    agent_params: dict[str, Any],
    bandit: GaussianBandit,
    num_rounds: int = 1000,
    run_name: str = None,
) -> dict[str, Any]:
    """Run a single agent experiment and return results."""

    # Initialize a new wandb run for this agent
    wandb.init(
        project="mh4521-bandit-comparison",
        name=run_name,
        config={
            "agent_type": agent_class.__name__,
            "bandit_arms": bandit.n_arms,
            "bandit_mean": bandit.mean,
            "bandit_std": bandit.std,
            "bandit_arms_std": bandit.arms_std,
            "num_rounds": num_rounds,
            "bandit_best_arm_mean": bandit.best_arm_mean,
            "bandit_arm_means": [arm.mean() for arm in bandit.arms],
            **agent_params,
        },
        tags=["comparison", agent_class.__name__.lower(), "multi_agent_run"],
    )

    # Create agent with bandit
    agent = agent_class(bandit, **agent_params)

    # Log bandit information
    wandb.log(
        {
            "bandit_best_arm_mean": bandit.best_arm_mean,
            "bandit_arm_means": [arm.mean() for arm in bandit.arms],
        }
    )

    # Run evaluation
    start_time = time.time()
    results = agent.evaluate(num_rounds=num_rounds, log_frequency=50)
    end_time = time.time()

    # Calculate performance metrics
    final_regret = results["regret"]
    final_reward = results["total_reward"]
    avg_regret = final_regret / num_rounds
    avg_reward = final_reward / num_rounds
    execution_time = end_time - start_time

    # Log final metrics
    wandb.log(
        {
            "final_cumulative_regret": final_regret,
            "final_cumulative_reward": final_reward,
            "final_average_regret": avg_regret,
            "final_average_reward": avg_reward,
            "execution_time_seconds": execution_time,
            "experiment_completed": True,
        }
    )

    # Create summary for comparison
    summary_results = {
        "agent_type": agent_class.__name__,
        "final_regret": final_regret,
        "final_reward": final_reward,
        "avg_regret": avg_regret,
        "avg_reward": avg_reward,
        "execution_time": execution_time,
        "cumulative_regret": results["cumulative_regret"],
        "cumulative_reward": results["cumulative_reward"],
    }

    wandb.finish()
    return summary_results


def compare_all_agents(
    num_rounds: int = 1000, bandit_config: dict[str, Any] = None, num_seeds: int = 3
):
    """Compare all agents across multiple seeds."""

    if bandit_config is None:
        bandit_config = {"n_arms": 4, "mean": 0, "std": 1, "arms_std": 0.1}

    # Define agent configurations
    agent_configs = [
        {
            "class": EpsAgent,
            "params": {"eps": 0.1, "alpha": 0.1},
            "name": "EpsilonGreedy",
        },
        {
            "class": EpsAgent,
            "params": {"eps": 0.05, "alpha": 0.1},
            "name": "EpsilonGreedy_low",
        },
        {"class": EtcAgent, "params": {"num_trials": 10}, "name": "ExploreThemCommit"},
        {
            "class": EtcAgent,
            "params": {"num_trials": 20},
            "name": "ExploreThemCommit_more",
        },
        {
            "class": UCBAgent,
            "params": {"delta": 0.1, "c": 2, "eps": 0.1},
            "name": "UCB",
        },
        {
            "class": UCBAgent,
            "params": {"delta": 0.05, "c": 1.5, "eps": 0.05},
            "name": "UCB_conservative",
        },
    ]

    all_results = []

    for seed in range(num_seeds):
        print(f"\n=== Running experiments with seed {seed} ===")

        # Create bandit with current seed
        bandit = GaussianBandit(seed=seed, **bandit_config)

        for config in agent_configs:
            agent_class = config["class"]
            agent_params = config["params"].copy()
            agent_name = config["name"]

            run_name = f"{agent_name}_seed_{seed}"
            print(f"Running {run_name}...")

            try:
                results = run_agent_experiment(
                    agent_class=agent_class,
                    agent_params=agent_params,
                    bandit=bandit,
                    num_rounds=num_rounds,
                    run_name=run_name,
                )
                results["seed"] = seed
                results["agent_name"] = agent_name
                all_results.append(results)

                print(f"  Final regret: {results['final_regret']:.4f}")
                print(f"  Average regret: {results['avg_regret']:.4f}")
                print(f"  Execution time: {results['execution_time']:.2f}s")

            except Exception as e:
                print(f"  Error running {run_name}: {e}")
                continue

    # Summarize results
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Group results by agent type
    agent_summaries = {}
    for result in all_results:
        agent_name = result["agent_name"]
        if agent_name not in agent_summaries:
            agent_summaries[agent_name] = {"regrets": [], "rewards": [], "times": []}

        agent_summaries[agent_name]["regrets"].append(result["avg_regret"])
        agent_summaries[agent_name]["rewards"].append(result["avg_reward"])
        agent_summaries[agent_name]["times"].append(result["execution_time"])

    # Print summary statistics
    for agent_name, summary in agent_summaries.items():
        regrets = np.array(summary["regrets"])
        rewards = np.array(summary["rewards"])
        times = np.array(summary["times"])

        print(f"\n{agent_name}:")
        print(f"  Average Regret: {regrets.mean():.4f} ± {regrets.std():.4f}")
        print(f"  Average Reward: {rewards.mean():.4f} ± {rewards.std():.4f}")
        print(f"  Execution Time: {times.mean():.2f}s ± {times.std():.2f}s")

    # Find best performing agent
    best_agent = min(agent_summaries.items(), key=lambda x: np.mean(x[1]["regrets"]))
    print(f"\nBest performing agent: {best_agent[0]}")
    print(f"  Average regret: {np.mean(best_agent[1]['regrets']):.4f}")

    return all_results, agent_summaries


if __name__ == "__main__":
    print("Starting multi-agent bandit comparison...")
    print("This will run multiple agents with different configurations.")
    print("Results will be logged to Weights & Biases.")

    # Run comprehensive comparison
    results, summaries = compare_all_agents(num_rounds=1000, num_seeds=3)

    print(
        "Comparison completed! Check your Weights & Biases dashboard for detailed "
        "results."
    )
    print("Project: mh4521-bandit-comparison")
