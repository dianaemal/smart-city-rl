from src.environment.city import CityServiceEnv
env = CityServiceEnv()
obs, info = env.reset()
print("Actual obs length:", len(obs))
print("Declared observation_space shape:", env.observation_space.shape)