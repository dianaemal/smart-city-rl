from src.environment.city_service_env import CityServiceEnv
from src.baselines.greedy import greedy_policy
import numpy as np

def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy(
            obs,
            env.num_crews,
            env.num_zones,
      
        )

        obs, reward, terminated, truncated, info = env.step(action)
       
       
        print(info)
        total_reward += reward

    return total_reward


if __name__ == "__main__":
   
    SEED = list(range(100))
    rewards = []
    for seed in SEED:
        env = CityServiceEnv()
        total_reward = run_episode(
            env,
            greedy_policy,
            seed
        )
        rewards.append(total_reward)

    
    print("Greedy mean reward:", np.mean(rewards))
    print("Greedy std reward:", np.std(rewards))