# MH4521 Lab 1: Multi-Armed Bandit Algorithms

This laboratory implements and compares different multi-armed bandit algorithms using Gaussian bandits. The project includes three main bandit strategies: Epsilon-Greedy, Explore-Then-Commit, and Upper Confidence Bound (UCB), with comprehensive experiment tracking via Weights & Biases.

## 📋 Overview

Multi-armed bandit problems model the trade-off between exploration and exploitation in sequential decision-making. This lab implements several classic algorithms to solve the bandit problem and provides tools for comparing their performance.

### Implemented Algorithms

1. **Epsilon-Greedy (ε-greedy)**: Explores randomly with probability ε, otherwise exploits the best known arm
2. **Explore-Then-Commit (ETC)**: Explores all arms equally for a fixed number of trials, then commits to the best arm
3. **Upper Confidence Bound (UCB)**: Uses confidence intervals to balance exploration and exploitation

## 🏗 Project Structure

```
lab1/
├── agent/                  # Agent implementations
│   ├── agent.py           # Base agent class
│   ├── eps_agent.py       # Epsilon-greedy agent
│   ├── etc_agent.py       # Explore-then-commit agent
│   └── ucb_agent.py       # UCB agent
├── arm/                   # Arm implementations
├── bandit/                # Bandit implementations
│   └── gaussian.py        # Gaussian bandit
├── compare_agents.py      # Multi-agent comparison script
├── run.py                 # Single agent experiment
├── setup_wandb.py         # Weights & Biases setup
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12

### Installation

1. Clone the repository and navigate to the lab1 directory
2. Install dependencies:
   ```bash
   uv sync
   ```

### Running Experiments

#### Single Agent Experiment

Run a single agent experiment with the UCB algorithm:

```bash
```

This script:
- Creates a 4-armed Gaussian bandit
- Runs a UCB agent for 1000 rounds
- Logs results to Weights & Biases
- Prints final performance metrics

#### Multi-Agent Comparison

Compare all implemented algorithms across multiple seeds:

```bash
python compare_agents.py
```

This comprehensive comparison:
- Tests 6 different agent configurations
- Runs each configuration across 3 different random seeds
- Logs detailed results to Weights & Biases
- Provides statistical summaries of performance

## 🔧 Configuration

### Agent Parameters

#### Epsilon-Greedy Agent
- `eps`: Exploration probability (0 ≤ eps ≤ 1)
- `alpha`: Learning rate for Q-value updates (0 < alpha < 1)

#### Explore-Then-Commit Agent
- `num_trials`: Number of exploration trials per arm

#### UCB Agent
- `delta`: Confidence parameter for UCB calculation
- `c`: Exploration constant
- `eps`: Optional random exploration probability (default: 0)

### Bandit Configuration

The Gaussian bandit supports:
- `n_arms`: Number of arms (default: 4)
- `mean`: Mean of the distribution for arm means (default: 0)
- `std`: Standard deviation for reward generation (default: 1)
- `arms_std`: Standard deviation for arm mean generation (default: 0.1)
- `seed`: Random seed for reproducibility

## 📊 Experiment Tracking

This project uses Weights & Biases for comprehensive experiment tracking:

### Logged Metrics

- **Per-round metrics**: Instant regret, cumulative regret, instant reward, cumulative reward, selected arm
- **Final metrics**: Total regret, average regret, total reward, average reward, execution time
- **Bandit information**: Arm means, best arm mean, configuration parameters

### Projects

- `mh4521-bandit-lab`: Single agent experiments
- `mh4521-bandit-comparison`: Multi-agent comparison experiments

## 📈 Performance Metrics

### Regret
- **Instant Regret**: Difference between optimal arm reward and selected arm reward
- **Cumulative Regret**: Sum of all instant regrets
- **Average Regret**: Cumulative regret divided by number of rounds

### Reward
- **Instant Reward**: Reward received from selected arm
- **Cumulative Reward**: Sum of all received rewards
- **Average Reward**: Cumulative reward divided by number of rounds

## 🎯 Expected Results

Based on the default configuration, you should expect:

1. **UCB** typically performs best with lowest regret
2. **Epsilon-Greedy** shows steady performance but may have higher regret
3. **Explore-Then-Commit** performance depends heavily on the exploration budget

The multi-agent comparison will provide statistical significance testing across multiple seeds.

## 🔄 Customization

### Adding New Agents

1. Inherit from the `Agent` base class
2. Implement the `__init__` and `play` methods
3. Add your agent to the comparison script

### Modifying Experiments

Edit the configuration dictionaries in `compare_agents.py` to:
- Change agent parameters
- Modify bandit settings
- Adjust number of rounds or seeds
- Add new agent configurations

## 🐛 Troubleshooting

### Weights & Biases Issues

If you encounter wandb issues:
1. Run `python setup_wandb.py` for guided setup
2. Set `WANDB_MODE=offline` for local tracking
3. Set `WANDB_MODE=disabled` to disable tracking

### Common Issues

- **Import errors**: Ensure you're running from the correct directory
- **Permission errors**: Check file permissions for wandb cache
- **Memory issues**: Reduce number of rounds or seeds for large experiments

## 📚 References

- [Multi-Armed Bandit Algorithms](https://banditalgs.com/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [UCB Algorithm Paper](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)

## 🤝 Contributing

This is a laboratory assignment for MH4521. Follow your course guidelines for submission and collaboration policies.
