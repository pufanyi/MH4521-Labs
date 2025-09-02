"""
Multi-Agent Comparison Script with Wandb Integration
This script runs multiple bandit agents and compares their performance using
Weights & Biases.
"""

import time
from typing import Any

import numpy as np
import wandb
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .agent.eps_agent import EpsAgent
from .agent.etc_agent import EtcAgent
from .agent.ucb_agent import UCBAgent
from .bandit.gaussian import GaussianBandit


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
        reinit=True,
        force=True,
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
    end_time = time.time() # This line was missing in the original string

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
    console = Console()

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
        {"class": EtcAgent, "params": {"num_trials": 10}, "name": "ExploreThenCommit"},
        {
            "class": EtcAgent,
            "params": {"num_trials": 20},
            "name": "ExploreThenCommit_more",
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

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
        console=console,
    )

    with Live(progress, refresh_per_second=10) as live:
        seed_task = progress.add_task("[green]Total Progress", total=num_seeds)

        for seed in range(num_seeds):
            progress.update(
                seed_task, description=f"[green]Running Seed {seed+1}/{num_seeds}"
            )

            live.console.print(
                Panel(
                    f"Running experiments with seed {seed}",
                    title="[bold blue]Seed Information[/bold blue]",
                )
            )

            bandit = GaussianBandit(seed=seed, **bandit_config)
            agent_task = progress.add_task(
                "[cyan]Agent Progress", total=len(agent_configs)
            )

            for config in agent_configs:
                agent_class = config["class"]
                agent_params = config["params"].copy()
                agent_name = config["name"]
                run_name = f"{agent_name}_seed_{seed}"

                progress.update(
                    agent_task, description=f"[cyan]Running: {agent_name}"
                )

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
                    live.console.print(f"  [green]✓[/green] {run_name} completed.")
                except Exception as e:
                    live.console.print(f"  [red]✗[/red] Error running {run_name}: {e}")
                finally:
                    progress.update(agent_task, advance=1)

            progress.remove_task(agent_task)
            progress.update(seed_task, advance=1)

    # Final summary section (outside the Live context)
    console.print(
        Panel(
            "Multi-Agent Experiment Summary",
            title="[bold green]Summary[/bold green]",
            expand=False,
        )
    )

    agent_summaries = {}
    for result in all_results:
        agent_name = result["agent_name"]
        if agent_name not in agent_summaries:
            agent_summaries[agent_name] = {"regrets": [], "rewards": [], "times": []}
        agent_summaries[agent_name]["regrets"].append(result["avg_regret"])
        agent_summaries[agent_name]["rewards"].append(result["avg_reward"])
        agent_summaries[agent_name]["times"].append(result["execution_time"])

    summary_table = Table(
        title="Agent Performance Summary", show_header=True, header_style="bold magenta"
    )
    summary_table.add_column("Agent Name", style="cyan", no_wrap=True)
    summary_table.add_column("Avg Regret", justify="right", style="green")
    summary_table.add_column("Avg Reward", justify="right", style="yellow")
    summary_table.add_column("Avg Time (s)", justify="right", style="blue")

    for agent_name, summary in agent_summaries.items():
        regrets = np.array(summary["regrets"])
        rewards = np.array(summary["rewards"])
        times = np.array(summary["times"])
        summary_table.add_row(
            agent_name,
            f"{regrets.mean():.4f} ± {regrets.std():.4f}",
            f"{rewards.mean():.4f} ± {rewards.std():.4f}",
            f"{times.mean():.2f} ± {times.std():.2f}",
        )
    console.print(summary_table)

    best_agent_name, best_agent_summary = min(
        agent_summaries.items(), key=lambda x: np.mean(x[1]["regrets"])
    )
    best_regret = np.mean(best_agent_summary["regrets"])
    console.print(
        Panel(
            f"Best performing agent: [bold cyan]{best_agent_name}[/bold cyan]\n" 
            f"  Average regret: [bold green]{best_regret:.4f}[/bold green]",
            title="[bold yellow]Top Performer[/bold yellow]",
        )
    )

    return all_results, agent_summaries


if __name__ == "__main__":
    console = Console()
    console.print(
        Panel(
            "Starting multi-agent bandit comparison...\n" 
            "This will run multiple agents with different configurations.\n" 
            "Results will be logged to Weights & Biases.",
            title="[bold blue]MH4521 Bandit Comparison[/bold blue]",
        )
    )

    results, summaries = compare_all_agents(num_rounds=1000, num_seeds=3)

    console.print(
        Panel(
            "Comparison completed! Check your Weights & Biases dashboard for detailed results.\n" 
            "Project: [bold]mh4521-bandit-comparison[/bold]",
            title="[bold green]Finished[/bold green]",
        )
    )