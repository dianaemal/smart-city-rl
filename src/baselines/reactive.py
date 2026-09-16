import numpy as np

def reactive_policy(obs, num_crews, num_zones):
    # return the indeces of the zones with highest backlog number
    backlog = obs[num_zones: num_zones * 2]
    return np.argpartition(backlog, -num_crews)[-num_crews:]

