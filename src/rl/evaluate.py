"""
Evaluation script for the score-vector action space.

Key difference from before: greedy now produces a zone LIST directly
(not a 22-dim score vector), so it's passed to env.step_with_zones(),
bypassing the score-interpretation logic entirely. PPO's output IS a
score vector, so it goes through the normal env.step() path.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.city import CityServiceEnv
from src.baselines.greedy2 import  greedy_policy  # <-- adjust to your actual import path

NUM_ZONES = 22
NUM_CREWS = 11
FORECAST_HORIZON = 5
N_EVAL_EPISODES = 100

EVAL_DATE_INDICES = np.load("results/eval_date_indices_scorevec.npy")

# ----------------------------------------------------------------
# PPO evaluation
# ----------------------------------------------------------------
def make_eval_env():
    return CityServiceEnv(date_indices=EVAL_DATE_INDICES, forecast_horizon=FORECAST_HORIZON)

raw_eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize.load(
    "results/models/vecnormalize_scorevec_final.pkl", raw_eval_env
)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load("results/models/best_scorevec_v2/best_model", env=eval_env)

ppo_rewards = []
for ep in range(N_EVAL_EPISODES):
    obs = eval_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        print(info)
        total_reward += reward[0]
    ppo_rewards.append(total_reward)

print("PPO mean reward:", np.mean(ppo_rewards))
print("PPO std reward:", np.std(ppo_rewards))

# ----------------------------------------------------------------
# Greedy evaluation — uses step_with_zones(), not step()
# ----------------------------------------------------------------
greedy_rewards = []
for start_idx in EVAL_DATE_INDICES[:N_EVAL_EPISODES]:
    env = CityServiceEnv(date_indices=np.array([start_idx]), forecast_horizon=FORECAST_HORIZON)
    obs, info = env.reset()
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        zone_list = greedy_policy(obs, NUM_CREWS, NUM_ZONES, crew_capacity=30,
                                   forecast_horizon=FORECAST_HORIZON)
        obs, reward, terminated, truncated, info = env.step_with_zones(zone_list)
  
        total_reward += reward
    greedy_rewards.append(total_reward)

print("Greedy mean reward:", np.mean(greedy_rewards))
print("Greedy std reward:", np.std(greedy_rewards))

print()
print(f"PPO vs Greedy gap: {np.mean(ppo_rewards) - np.mean(greedy_rewards):.1f}")