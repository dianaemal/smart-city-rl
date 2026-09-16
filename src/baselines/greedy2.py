import numpy as np


def greedy_policy(
    obs,
    num_crews,
    num_zones,
    crew_capacity=30,
    forecast_horizon=5,
):
    """
    Returns a list of `num_crews` target zone indices (always distinct),
    chosen by the same magnitude-only reactive rule as before: send each
    crew to whichever zone has the highest remaining (backlog + today's
    demand), recomputing after each pick to avoid double-counting.

    Greedy is intentionally kept reactive — it only reads TODAY's demand
    and today's backlog. It does NOT see future forecast days or
    request_age, since the entire point of this comparison is that
    greedy represents the current reactive-only baseline.

    NOTE: this now returns a zone LIST, not an env action. Pass it to
    env.step_with_zones(zone_list) directly, not env.step(action).
    """
    # obs layout: [forecast_window (horizon*num_zones), backlog (num_zones),
    #              request_age (num_zones), crew_locations (num_crews)]
    today_demand = obs[:num_zones]

    backlog_start = num_zones * forecast_horizon
    backlog = obs[backlog_start: backlog_start + num_zones]
    backlog = np.expm1(backlog)  # undo the log1p transform from the env

    remaining_need = backlog + today_demand
    zone_list = []
    chosen_zones = set()

    for _ in range(num_crews):
        scores = remaining_need.copy()
        for zone in chosen_zones:
            scores[zone] = -np.inf
        best_zone = int(np.argmax(scores))
        zone_list.append(best_zone)
        chosen_zones.add(best_zone)
        remaining_need[best_zone] -= crew_capacity
        remaining_need = np.maximum(remaining_need, 0)

    return np.array(zone_list, dtype=np.int32)