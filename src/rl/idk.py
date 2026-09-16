"""
Regenerates train_date_indices / eval_date_indices fresh, using the
CURRENT environment's actual episode_length and forecast_horizon —
avoids any mismatch with stale .npy files from earlier script versions.
"""

import numpy as np
from src.environment.city import CityServiceEnv  # adjust if your path differs
from src.forecasting.prediction_loader import PredictionLoader
from src.rl.date_utils import split_date_indices

# Instantiate a throwaway env just to read its actual parameters
probe_env = CityServiceEnv()
episode_length = probe_env.episode_length
forecast_horizon = probe_env.forecast_horizon

predictor = PredictionLoader()
num_dates = len(predictor.available_dates)

print(f"num_dates={num_dates}, episode_length={episode_length}, forecast_horizon={forecast_horizon}")

train_starts, eval_starts = split_date_indices(
    num_dates=num_dates,
    episode_length=episode_length,
    forecast_horizon=forecast_horizon,
    holdout_fraction=0.15,
)

# Sanity check against the env's own constraint
max_start = num_dates - episode_length - forecast_horizon + 1
print(f"env's own max_start = {max_start}")
print(f"train_starts max = {train_starts.max()}, eval_starts max = {eval_starts.max()}")

assert train_starts.max() <= max_start, "train_starts still exceeds env's max_start!"
assert eval_starts.max() <= max_start, "eval_starts still exceeds env's max_start!"

np.save("results/train_date_indices_scorevec.npy", train_starts)
np.save("results/eval_date_indices_scorevec.npy", eval_starts)

print("Regenerated and verified. Saved to results/train_date_indices_scorevec.npy")
print("and results/eval_date_indices_scorevec.npy")