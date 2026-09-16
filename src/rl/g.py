import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.city_service_env import CityServiceEnv
from src.baselines.greedy2 import greedy_policy  # <-- adjust to your actual import path

NUM_ZONES = 22
NUM_CREWS = 11
N_EVAL_EPISODES = 100

EVAL_DATE_INDICES = np.load("results/eval_date_indices.npy")
greedy_rewards = []
for start_idx in EVAL_DATE_INDICES[:N_EVAL_EPISODES]:
    env = CityServiceEnv(date_indices=np.array([start_idx]))
    obs, info = env.reset()
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = greedy_policy(obs, NUM_CREWS, NUM_ZONES, crew_capacity=30)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    greedy_rewards.append(total_reward)

print("Greedy mean reward:", np.mean(greedy_rewards))
print("Greedy std reward:", np.std(greedy_rewards))

print()
