import numpy as np
from stable_baselines3 import PPO
from src.environment.city_service_env import CityServiceEnv

model = PPO.load("results/models/ppo_city_service_v8")

SEEDS = list(range(100))
rewards = []

for seed in SEEDS:
    env = CityServiceEnv()
    obs, info = env.reset(seed=seed)
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(info)

    rewards.append(total_reward)
print(rewards[:10])
print("PPO mean reward:", np.mean(rewards))
print("PPO std reward:", np.std(rewards))