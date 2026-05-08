# Model-free Learning in Markov Decision Processes with Exogenous Signals

This project explores MDPs with Exogenous Inputs (Exo-MDPs), a class of sequential decision-making problems where an agent's actions affect only the endogenous part of the state space, while the exogenous part evolves independently. This distinction allow us to improve learning guarantees and enhance sample efficiency, under the assumption of knowing the controllable transition model.
The framework provides implementations of several state-of-the-art reinforcement learning algorithms and evaluation tools.

## Environments

The project includes three distinct environments:

### 1. **Elevator Simulation** (`exomdp/elevator/`)

A multi-floor elevator scheduling environment where the agent must manage elevator movements to minimize passenger waiting time.

- **State**: Elevator position, number of passengers on board, waiting queue at each floor, arrivals queue at each floor
- **Actions**: Move up, stay, move down
- **Dynamics**: Stochastic passenger arrivals following configurable distributions
- **Variants**: Standard world and tiny world configurations with variable arrival rates

### 2. **Taxi Domain** (`exomdp/taxi/`)

A grid-world taxi dispatch problem based on the classic Taxi-v3 environment.

- **State**: Taxi position, passenger location, destination location, traffic
- **Actions**: Move north/south/east/west, pickup, dropoff
- **Goal**: Pick up passengers and drop them at their destinations efficiently

### 3. **Trading Environment** (`exomdp/trading/`)

An algorithmic trading environment for learning optimal execution strategies.

- **State**: Current price, portfolio holdings
- **Actions**: Buy, sell, or hold
- **Goal**: Liquidating the position in optimal way

## Algorithms

The framework implements the following reinforcement learning algorithms:

| Algorithm                             | File                | Type            | Description                                                 |
| ------------------------------------- | ------------------- | --------------- | ----------------------------------------------------------- |
| **Q-Learning**                        | `algo/ql.py`        | Tabular         | Classic value-iteration method for discrete spaces          |
| **Exogenous-Aware Q-Learning (EXAQ)** | `algo/exaq.py`      | Tabular         | Q-Learning exploiting exogenous information                 |
| **UCBVI**                             | `algo/ucbvi.py`     | Tabular         | Upper Confidence Bound Value Iteration                      |
| **PTO**                               | `algo/pto.py`       | Tabular         | Value iteration without exploration bonuses                 |
| **PPO**                               | `algo/ppo.py`       | Policy Gradient | Proximal Policy Optimization for continuous/complex domains |
| **Baselines**                         | `algo/baselines.py` | Scripted        | Hand-crafted policies for comparison                        |

## Installation

### Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Daveonwave/Exo-MDP.git
cd Exo-MDP
```

2. Create and activate a conda environment:

```bash
conda create -n exomdp python=3.12
conda activate exomdp
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Training Experiments

#### Command-Line Interface

Run training with the main script:

```bash
python main.py \
    --env elevator \
    --env_id elevator-v0 \
    --algo pto \
    --exp_name <exp-name> \
    --dest_folder <dest-folder> \
    --world "world.yaml" \
    --n_episodes 10000 \
    --gamma 1 \
    --eval_episodes 50 \
    --eval_every 1 \
    --train_seeds 1 2 3 4 5 \
    --eval_seed 1234
```

#### Key Arguments

- `--env`: Environment type (`elevator`, `taxi`, `trading`)
- `--env_id`: Gymnasium environment ID
- `--algo`: Algorithm to use (`ql`, `exaq`, `ucbvi`, `pto`, `ppo`)
- `--exp_name`: Experiment name for logging
- `--n_episodes`: Number of training episodes
- `--n_seeds`: Number of random seeds to run (default: 1)
- `--eval_every`: Evaluation frequency (episodes)
- `--eval_episodes`: Episodes per evaluation
- `--dest_folder`: Output directory for logs and models
