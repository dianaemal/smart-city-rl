from src.environment.city_service_env import CityServiceEnv





def run_random_episode(env):
    obs, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(info)

    return total_reward

if __name__ == "__main__":
    env = CityServiceEnv()

    total_reward = run_random_episode(
        env,
       
    )

    print("Random total reward:", total_reward)