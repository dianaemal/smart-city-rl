"""
Evaluates PPO against THREE baselines on the same held-out dates:
  1. Magnitude-based greedy (your original baseline)
  2. Reactive Dispatch (FCFS, no proactive movement)
  3. Greedy Heuristic (oldest-or-closest)

All four are evaluated on identical held-out date seeds for a fair,
direct comparison table.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.city import CityServiceEnv  # adjust path if needed
from src.environment.travel import TRAVEL_MATRIX
from src.baselines.greedy2 import greedy_policy
from src.baselines.baseline_policies import reactive_fcfs_policy, closest_or_oldest_policy

NUM_ZONES = 22
NUM_CREWS = 11
FORECAST_HORIZON = 5
N_EVAL_EPISODES = 100

EVAL_DATE_INDICES = np.load("results/eval_date_indices_scorevec.npy")


def run_baseline(policy_fn, name):
    rewards = []
    for start_idx in EVAL_DATE_INDICES[:N_EVAL_EPISODES]:
        env = CityServiceEnv(date_indices=np.array([start_idx]), forecast_horizon=FORECAST_HORIZON)
        obs, info = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            zone_list = policy_fn(obs, env.crew_locations)
            obs, reward, terminated, truncated, info = env.step_with_zones(zone_list)
            total_reward += reward
        rewards.append(total_reward)
    print(f"{name}: mean={np.mean(rewards):.1f}  std={np.std(rewards):.1f}")
    return np.array(rewards)


# ----------------------------------------------------------------
# PPO
# ----------------------------------------------------------------
def make_eval_env():
    return CityServiceEnv(date_indices=EVAL_DATE_INDICES, forecast_horizon=FORECAST_HORIZON)

raw_eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize.load("results/models/vecnormalize_scorevec_v2_final.pkl", raw_eval_env)
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
        total_reward += reward[0]
    ppo_rewards.append(total_reward)

print(f"PPO: mean={np.mean(ppo_rewards):.1f}  std={np.std(ppo_rewards):.1f}")

# ----------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------
greedy_rewards = run_baseline(
    lambda obs, crew_loc: greedy_policy(obs, NUM_CREWS, NUM_ZONES, crew_capacity=30,
                                         forecast_horizon=FORECAST_HORIZON),
    "Magnitude-Greedy"
)

fcfs_rewards = run_baseline(
    lambda obs, crew_loc: reactive_fcfs_policy(obs, NUM_CREWS, NUM_ZONES, crew_loc,
                                                forecast_horizon=FORECAST_HORIZON),
    "Reactive-FCFS"
)

closest_oldest_rewards = run_baseline(
    lambda obs, crew_loc: closest_or_oldest_policy(obs, NUM_CREWS, NUM_ZONES, crew_loc,
                                                     TRAVEL_MATRIX, forecast_horizon=FORECAST_HORIZON),
    "Oldest-or-Closest"
)

# ----------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------
print()
print(f"{'Policy':<20}{'Mean':>10}{'Std':>10}{'Gap vs PPO':>14}")
print("-" * 54)
ppo_mean = np.mean(ppo_rewards)
for name, rewards in [
    ("PPO", ppo_rewards),
    ("Magnitude-Greedy", greedy_rewards),
    ("Reactive-FCFS", fcfs_rewards),
    ("Oldest-or-Closest", closest_oldest_rewards),
]:
    mean = np.mean(rewards)
    std = np.std(rewards)
    gap = ppo_mean - mean
    print(f"{name:<20}{mean:>10.1f}{std:>10.1f}{gap:>14.1f}")