import numpy as np


def greedy_policy(
    obs,
    num_crews,
    num_zones,
    crew_capacity=30,
    allow_duplicate_zones=False,
):
    demand = obs[:num_zones]
    backlog = obs[num_zones:num_zones * 2]

    remaining_need = backlog + demand

    actions = []
    chosen_zones = set()

    for _ in range(num_crews):
        scores = remaining_need.copy()

        if not allow_duplicate_zones:
            for zone in chosen_zones:
                scores[zone] = -np.inf

        best_zone = int(np.argmax(scores))

        actions.append(best_zone)
        chosen_zones.add(best_zone)

        remaining_need[best_zone] -= crew_capacity
        remaining_need = np.maximum(remaining_need, 0)

    return np.array(actions, dtype=np.int32)