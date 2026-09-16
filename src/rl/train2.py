"""
Continues training from the current best score-vector checkpoint, now
with the age_pressure log-fix in city_service_env.py already applied.
Same observation/action space as before -> safe to resume from checkpoint.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.city import CityServiceEnv  # adjust path if needed

FORECAST_HORIZON = 5
ADDITIONAL_TIMESTEPS = 400_000
CHECKPOINT_EVERY = 50_000
EVAL_EVERY = 25_000
N_EVAL_EPISODES = 60   # larger than before, for a more stable read on the gap

train_starts = np.load("results/train_date_indices_scorevec.npy")
eval_starts = np.load("results/eval_date_indices_scorevec.npy")

def make_train_env():
    env = CityServiceEnv(date_indices=train_starts, forecast_horizon=FORECAST_HORIZON)
    return Monitor(env)

def make_eval_env():
    env = CityServiceEnv(date_indices=eval_starts, forecast_horizon=FORECAST_HORIZON)
    return Monitor(env)

train_env = DummyVecEnv([make_train_env])
train_env = VecNormalize.load("results/models/vecnormalize_scorevec_final.pkl", train_env)
train_env.training = True
train_env.norm_reward = True

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize.load("results/models/vecnormalize_scorevec_final.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load(
    "results/models/best_scorevec/best_model",
    env=train_env,
)

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_EVERY,
    save_path="results/models/checkpoints_scorevec_v2/",
    name_prefix="ppo_city_service_scorevec_v2",
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    eval_env,
    eval_freq=EVAL_EVERY,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    log_path="results/tb_logs_scorevec/eval_v2/",
    best_model_save_path="results/models/best_scorevec_v2/",
)

model.learn(
    total_timesteps=ADDITIONAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
    reset_num_timesteps=False,
)

model.save("results/models/ppo_city_service_scorevec_v2_final")
train_env.save("results/models/vecnormalize_scorevec_v2_final.pkl")

print("Continued training complete.")
print("Best checkpoint: results/models/best_scorevec_v2/best_model.zip")
print("Watch eval/mean_reward in results/tb_logs_scorevec/eval_v2/ vs greedy's ~164.5")