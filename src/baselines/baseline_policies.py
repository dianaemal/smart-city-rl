import numpy as np


def reactive_fcfs_policy(obs, num_crews, num_zones, crew_locations, forecast_horizon=5):
    """
    Reactive Dispatch (FCFS, no proactive movement):
    - Crews currently sitting in a zone with active backlog STAY there
      (no repositioning while there's still work where they are).
    - Crews currently in a zone with zero backlog are "free" and get
      reassigned to whichever zone has been waiting LONGEST (oldest
      request_age), processed in queue order (oldest first).
    - No use of magnitude, distance, or forecast at all - purely "serve
      whoever's been waiting longest, don't move unless idle."

    Duplicates are allowed and expected here (multiple crews can already
    be sitting in the same active zone) - this baseline does not
    optimize coverage, intentionally.
    """
    backlog_start = num_zones * forecast_horizon
    backlog = obs[backlog_start: backlog_start + num_zones]
    backlog = np.expm1(backlog)  # undo log1p

    age_start = backlog_start + num_zones
    request_age = obs[age_start: age_start + num_zones]

    zone_list = np.zeros(num_crews, dtype=np.int32)
    idle_crew_indices = []

    for crew_idx in range(num_crews):
        current_zone = int(crew_locations[crew_idx])
        if backlog[current_zone] > 0:
            zone_list[crew_idx] = current_zone  # stay, no proactive movement
        else:
            idle_crew_indices.append(crew_idx)

    if idle_crew_indices:
        # queue order = oldest request_age first
        queue_order = np.argsort(-request_age)
        q_ptr = 0
        for crew_idx in idle_crew_indices:
            # find next zone in queue order with any backlog at all
            while q_ptr < num_zones and backlog[queue_order[q_ptr]] <= 0:
                q_ptr += 1
            if q_ptr < num_zones:
                zone_list[crew_idx] = queue_order[q_ptr]
            else:
                zone_list[crew_idx] = current_zone  # no active requests left, stay put

    return zone_list


def closest_or_oldest_policy(obs, num_crews, num_zones, crew_locations,
                              travel_matrix, forecast_horizon=5):
    """
    Greedy Heuristic (oldest-or-closest):
    For each crew, rank all zones with active backlog by request_age
    (oldest first); break ties by travel distance from that crew's
    current location (closest first). Processes crews in order, removing
    each chosen zone from the pool so assignments stay distinct.
    """
    backlog_start = num_zones * forecast_horizon
    backlog = obs[backlog_start: backlog_start + num_zones]
    backlog = np.expm1(backlog)

    age_start = backlog_start + num_zones
    request_age = obs[age_start: age_start + num_zones]

    zone_list = np.zeros(num_crews, dtype=np.int32)
    available_zones = set(z for z in range(num_zones) if backlog[z] > 0)

    for crew_idx in range(num_crews):
        if not available_zones:
            zone_list[crew_idx] = crew_locations[crew_idx]
            continue

        current_zone = int(crew_locations[crew_idx])
        # rank candidates: primary key = -age (oldest first), secondary = distance
        candidates = sorted(
            available_zones,
            key=lambda z: (-request_age[z], travel_matrix[current_zone, z])
        )
        chosen = candidates[0]
        zone_list[crew_idx] = chosen
        available_zones.discard(chosen)

    return zone_list