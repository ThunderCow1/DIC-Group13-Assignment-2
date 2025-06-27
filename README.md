# DIC Group 13 Assignment 2: Continuous Environment RL Agents

This project implements and evaluates reinforcement learning agents (DQN, PPO) in a continuous 2D environment for the Data Intelligence Challenge.

## Project Structure

```
.
├── agents/                # RL agent implementations (DQN, PPO, etc.)
├── environment/           # Environment, robot, map, and helper functions
├── maps/                  # Map JSON files
├── networks/              # Saved model checkpoints
├── results_DQN.json       # DQN evaluation results
├── results_PPO.json       # PPO evaluation results
├── evaluate.py            # Main script for training/evaluating agents
└── ...
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- pygame
- tqdm

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Usage

### Training and Evaluating Agents

Run the main evaluation script:

```sh
python evaluate.py maps/map1.json --agent DQN --train 2500 --evaluation 10 --Steps 1000 --Draw True --val_times 50
```

**Arguments:**
- `MAP`: Path to the map JSON file (e.g., `maps\map2.json`)
- `--agent`: Agent type (`human`, `intuitive`, `DQN`, `PPO`)
- `--train`: Number of training iterations (default: 2500)
- `--evaluation`: Number of evaluation episodes (default: 10)
- `--Steps`: Max steps per episode (default: 1000)
- `--Draw`: Whether to render the environment (default: True)
- `--val_times`: Number of validation cycles (default: 50)

### Example: PPO Agent

```sh
python evaluate.py maps\map2.json --agent PPO --train 2500 --evaluation 10 --Steps 1000 --Draw True --val_times 50
```

### Human/Intuitive Agent

For manual or intuitive agent control:

```sh
python evaluate.py maps/map2.json --agent intuitive
```

## Output

- Trained models are saved in the `networks/` directory.
- Evaluation results are saved as `results_DQN.json` or `results_PPO.json`.

## Code Overview

- [`evaluate.py`](evaluate.py): Main entry point for training and evaluation.
- [`agents/DQN_torch2.py`](agents/DQN_torch2.py): DQN agent implementation.
- [`agents/PPO_agent2.py`](agents/PPO_agent2.py): PPO agent implementation.
- [`environment/environment.py`](environment/environment.py): Environment logic.
- [`environment/robot.py`](environment/robot.py): Robot agent logic.
- [`environment/map.py`](environment/map.py): Map and obstacle logic.

## Visualization

Use the plotting scripts to visualize mean cumulative rewards:

```sh
python plot_rewards.py
```

## License

This project is for educational purposes.

---

For more details, see the code and comments in each file.