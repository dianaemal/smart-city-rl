# AI-Driven Smart City Service Management

Reinforcement-learning system for dispatching city service crews across zones in Vancouver, BC, using demand forecasting and a PPO policy trained against multiple baseline heuristics.

> **Note:** this repo predates a rename — despite the `LLM_Research` name, it does not use LLMs. It's a demand forecasting + RL scheduling project. (Renaming the GitHub repo itself, e.g. to `smart-city-rl`, is recommended.)

## Overview

The pipeline has three stages:

1. **Forecasting** — an XGBoost model predicts service-request demand per zone, per day, from historical request data.
2. **Environment** — `CityServiceEnv` (Gymnasium-compatible) simulates crews being dispatched across zones based on forecasted and actual demand, travel time between zones, and a backlog/utilization reward signal.
3. **Policy training & evaluation** — a PPO agent (Stable-Baselines3) is trained on the environment and evaluated against three baseline dispatch policies: reactive (FCFS), greedy (closest-or-oldest), and a magnitude-based greedy heuristic.

## Project structure

```
src/
  forecasting/     # demand prediction (XGBoost) — training, loading, generating predictions
  environment/      # CityServiceEnv: the Gymnasium environment, travel-time matrix, zone config
  baselines/        # non-RL dispatch policies used as comparison points
  rl/               # PPO training, evaluation, and results/graph generation

notebooks/
  demand_analysis.ipynb, exploration.ipynb, exploration2.ipynb, forecasting.ipynb
  figures/          # exported plots from the above

results/
  models/           # final trained PPO models + the demand forecasting model
  graphs/           # training curves, reward distributions, backlog/utilization trajectories
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running

```bash
python -m src.forecasting.demand_predictor    # train the demand model
python -m src.rl.train_ppo                    # train the PPO dispatch policy
python -m src.rl.final                        # evaluate PPO against all baselines
python -m src.rl.generate_graphs               # regenerate the plots in results/graphs/
```

## Results

Training curves, reward distributions, and per-episode backlog/utilization trajectories comparing PPO against the baseline policies are in `results/graphs/`.

## Status / known limitations

- Baseline comparison currently covers reactive, greedy, and magnitude-greedy policies; no non-RL optimization-based baseline (e.g. MILP) yet.
- `results/models/` keeps only the final trained models; intermediate training checkpoints are not tracked.
