"""
Training script for the restructured CityServiceEnv:
- Continuous 22-dim priority-score action space (replaces the old
  MultiDiscrete([22]*11)), with a deterministic top-k + nearest-crew
  assignment happening inside the env itself.
- Forecast horizon, request_age, and travel cost are part of the
  observation/reward as already implemented in city_service_env.py.
- Held-out date split accounts for forecast_horizon (no leakage).
- VecNormalize for reward scaling, EvalCallback for deterministic
  tracking on held-out dates from step one (this was missing in earlier
  runs and is what let the v5 regression go undetected).

This is a from-scratch run — the new action space is not compatible
with any previous checkpoint.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.city import CityServiceEnv
from src.forecasting.prediction_loader import PredictionLoader
from src.rl.date_utils import split_date_indices

EPISODE_LENGTH = 30
FORECAST_HORIZON = 5
TOTAL_TIMESTEPS = 600_000
CHECKPOINT_EVERY = 50_000
EVAL_EVERY = 25_000

os.makedirs("results/models/checkpoints_scorevec", exist_ok=True)
os.makedirs("results/tb_logs_scorevec", exist_ok=True)

# ----------------------------------------------------------------
# 1. Train/eval date split, accounting for forecast_horizon
# ----------------------------------------------------------------
predictor = PredictionLoader()
num_dates = len(predictor.available_dates)

train_starts, eval_starts = split_date_indices(
    num_dates=num_dates,
    episode_length=EPISODE_LENGTH,
    forecast_horizon=FORECAST_HORIZON,
    holdout_fraction=0.15,
)
print(f"Train start-date pool: {len(train_starts)} options")
print(f"Eval (held-out) start-date pool: {len(eval_starts)} options")

np.save("results/train_date_indices_scorevec.npy", train_starts)
np.save("results/eval_date_indices_scorevec.npy", eval_starts)

# ----------------------------------------------------------------
# 2. Training env
# ----------------------------------------------------------------
def make_train_env():
    env = CityServiceEnv(date_indices=train_starts, forecast_horizon=FORECAST_HORIZON)
    env = Monitor(env)
    return env

def make_eval_env():
    env = CityServiceEnv(date_indices=eval_starts, forecast_horizon=FORECAST_HORIZON)
    env = Monitor(env)
    return env

train_env = DummyVecEnv([make_train_env])
train_env = VecNormalize(
    train_env,
    norm_obs=False,      # obs already log1p-transformed manually in the env
    norm_reward=True,
    clip_reward=10.0,
    gamma=0.99,
)

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize(
    eval_env,
    norm_obs=False,
    norm_reward=False,   # report RAW reward for deterministic tracking
    gamma=0.99,
)
eval_env.training = False

# ----------------------------------------------------------------
# 3. PPO model — continuous action space now uses a Gaussian policy
# ----------------------------------------------------------------
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    n_steps=512,
    batch_size=128,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,     # continuous action entropy behaves differently than
                       # discrete; start moderate, watch the eval curve
    tensorboard_log="results/tb_logs_scorevec/",
)

# ----------------------------------------------------------------
# 4. Callbacks — checkpointing + deterministic eval tracking FROM STEP 1
# ----------------------------------------------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_EVERY,
    save_path="results/models/checkpoints_scorevec/",
    name_prefix="ppo_city_service_scorevec",
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    eval_env,
    eval_freq=EVAL_EVERY,
    n_eval_episodes=40,
    deterministic=True,
    log_path="results/tb_logs_scorevec/eval/",
    best_model_save_path="results/models/best_scorevec/",
)

# ----------------------------------------------------------------
# 5. Train
# ----------------------------------------------------------------
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
)

model.save("results/models/ppo_city_service_scorevec_final")
train_env.save("results/models/vecnormalize_scorevec_final.pkl")

print("Training complete.")
print("Final model: results/models/ppo_city_service_scorevec_final")
print("Best deterministic checkpoint: results/models/best_scorevec/best_model.zip")
print()
print("Watch BOTH in TensorBoard:")
print("  rollout/ep_rew_mean        (stochastic, continuous logging)")
print("  eval/mean_reward           (deterministic, held-out, every 25k steps)")