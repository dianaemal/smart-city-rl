import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.environment.data.zone_config import LOCAL_AREAS
from src.forecasting.prediction_loader import PredictionLoader
from src.environment.travel import TRAVEL_MATRIX


class CityServiceEnv(gym.Env):
    """
    Action space redesign:

    Instead of the policy directly choosing 11 zone indices (a 22^11
    joint space with no built-in coordination between crews), the policy
    now outputs a single 22-dimensional PRIORITY SCORE vector — one score
    per zone, representing "how urgently does this zone need a crew
    today." A deterministic assignment step (identical in spirit to how
    greedy already works) converts that score vector into actual crew
    assignments:

      1. Take the top `num_crews` zones by score (always distinct,
         exactly like greedy's no-duplicate rule) -> guarantees full
         22-zone, 11-crew realism with no possibility of the
         duplicate-zone collapse that caused most of the earlier
         training regressions.
      2. Match physical crews to those target zones via a simple nearest-
         crew heuristic (minimize travel cost), since deciding WHICH
         physical crew goes WHERE (for travel-cost purposes) is a much
         easier sub-problem than learning it end-to-end.

    This keeps all 22 zones and 11 crews — nothing is shrunk — but turns
    the learning problem into "learn a 22-dim scoring function" instead
    of "learn an 11-tuple from a combinatorially huge joint space."
    """

    def __init__(self, date_indices=None, forecast_horizon=5):
        self.num_crews = 11
        self.num_zones = len(LOCAL_AREAS)
        self.crew_capacity = 30
        self.episode_length = 30
        self.forecast_horizon = forecast_horizon
        self.current_day = 0
        self.forecast_window = np.zeros((self.forecast_horizon, self.num_zones))
        self.backlog = np.zeros(self.num_zones)
        self.crew_locations = np.zeros(self.num_crews, dtype=int)
        self.predictor = PredictionLoader()
        self.request_age = np.zeros(self.num_zones, dtype=np.float32)

        self.date_indices = date_indices

        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(
                self.num_zones * self.forecast_horizon
                + self.num_zones
                + self.num_zones
                + self.num_crews,
            ),
            dtype=np.float32
        )

        # NEW: continuous priority-score vector, one score per zone
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_zones,),
            dtype=np.float32
        )

    def _get_obs(self):
        obs = np.concatenate((
            self.forecast_window.flatten().astype(np.float32),
            np.log1p(self.backlog).astype(np.float32),
            self.request_age.astype(np.float32),
            self.crew_locations.astype(np.float32)
        ))
        return obs.astype(np.float32)

    def _load_forecast_window(self, date_idx):
        last_idx = len(self.predictor.available_dates) - 1
        window = np.zeros((self.forecast_horizon, self.num_zones), dtype=np.float32)
        for i in range(self.forecast_horizon):
            idx = min(date_idx + i, last_idx)
            date = self.predictor.available_dates[idx]
            window[i] = self.predictor.get_prediction_for_date(date)
        return window

    def _sample_start_date_idx(self):
        max_start = (
            len(self.predictor.available_dates)
            - self.episode_length
            - self.forecast_horizon
            + 1
        )

        if max_start < 0:
            raise ValueError("Not enough dates for episode_length + forecast_horizon.")

        if self.date_indices is not None:
            valid = self.date_indices[self.date_indices <= max_start]
            if len(valid) == 0:
                raise ValueError(
                    f"No valid date indices. max_start={max_start}, "
                    f"min date_index={self.date_indices.min()}, "
                    f"max date_index={self.date_indices.max()}"
                )
            return int(self.np_random.choice(valid))

        return int(self.np_random.integers(0, max_start + 1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        self.request_age = np.zeros(self.num_zones, dtype=np.float32)

        self.start_date_idx = self._sample_start_date_idx()
        self.forecast_window = self._load_forecast_window(self.start_date_idx)

        self.crew_locations = self.np_random.integers(
            low=0, high=self.num_zones, size=self.num_crews
        ).astype(np.int32)

        self.backlog = self.np_random.integers(
            low=0, high=10, size=self.num_zones
        ).astype(np.float32)

        return self._get_obs(), {}

    def _scores_to_zone_list(self, scores):
        """
        Top-`num_crews` zones by score, always distinct (mirrors greedy's
        no-duplicate convention, and structurally guarantees full
        coverage diversity regardless of how confident/peaked the
        policy's scores are).
        """
        # argsort ascending -> take last num_crews -> reverse for
        # highest-first order (order matters for crew-matching priority)
        top_idx = np.argsort(scores)[-self.num_crews:][::-1]
        return top_idx.astype(np.int32)

    def _match_crews_to_zones(self, zone_list, old_locations):
        """
        Assigns physical crews to the chosen target zones to minimize
        total travel cost, via a simple greedy nearest-crew heuristic:
        for each target zone (in priority order), pick whichever
        still-unassigned crew has the lowest travel cost to get there.
        This is a reasonable heuristic, not a globally optimal assignment
        (true optimal would be a full bipartite matching / Hungarian
        algorithm) — chosen deliberately because crew-to-zone routing is
        a much easier sub-problem than the prioritization decision, and
        doesn't need to be learned end-to-end by the policy.
        """
        unassigned_crews = list(range(self.num_crews))
        assignment = np.zeros(self.num_crews, dtype=np.int32)

        for target_zone in zone_list:
            best_crew = None
            best_cost = None
            for crew_idx in unassigned_crews:
                cost = TRAVEL_MATRIX[old_locations[crew_idx], target_zone]
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_crew = crew_idx
            assignment[best_crew] = target_zone
            unassigned_crews.remove(best_crew)

        return assignment

    def step(self, action):
        scores = np.array(action, dtype=np.float32)
        zone_list = self._scores_to_zone_list(scores)
        return self._step_core(zone_list)

    def step_with_zones(self, zone_list):
        """
        Bypass the score interpretation entirely — used by baseline
        policies (e.g. greedy) that already decide on a list of target
        zones directly, rather than a learned score vector.
        """
        zone_list = np.array(zone_list, dtype=np.int32)
        return self._step_core(zone_list)

    def _step_core(self, zone_list):
        old_locations = self.crew_locations.copy()

        # --------------------------------------------------
        # 1. Match physical crews to the chosen target zones
        # --------------------------------------------------
        action = self._match_crews_to_zones(zone_list, old_locations)
        self.crew_locations = action

        # --------------------------------------------------
        # 2. Travel cost
        # --------------------------------------------------
        travel_cost = 0.0
        for crew_idx in range(self.num_crews):
            start_zone = old_locations[crew_idx]
            end_zone = action[crew_idx]
            travel_cost += TRAVEL_MATRIX[start_zone, end_zone]

        # --------------------------------------------------
        # 3. Today's actual requests arrive
        # --------------------------------------------------
        new_requests = self.np_random.normal(loc=self.forecast_window[0], scale=5)
        new_requests = np.round(new_requests)
        new_requests = np.maximum(0, new_requests).astype(np.float32)
        self.backlog += new_requests

        backlog_before_service = self.backlog.copy()

        # --------------------------------------------------
        # 4. Service capacity (always 11 distinct zones now)
        # --------------------------------------------------
        service_capacity = np.zeros(self.num_zones, dtype=np.float32)
        for zone in zone_list:
            service_capacity[zone] += self.crew_capacity

        # --------------------------------------------------
        # 5. Resolve requests
        # --------------------------------------------------
        resolved = np.minimum(self.backlog, service_capacity)
        self.backlog -= resolved

        # --------------------------------------------------
        # 6. Update request age
        # --------------------------------------------------
        self.request_age[self.backlog > 0] += 1
        self.request_age[self.backlog == 0] = 0

        served_fraction = np.zeros(self.num_zones, dtype=np.float32)
        nonzero_mask = backlog_before_service > 0
        served_fraction[nonzero_mask] = (
            resolved[nonzero_mask] / backlog_before_service[nonzero_mask]
        )
        self.request_age *= (1.0 - served_fraction)

        # --------------------------------------------------
        # 7. Metrics
        # --------------------------------------------------
        total_resolved = float(np.sum(resolved))
        total_capacity = float(np.sum(service_capacity))
        wasted_capacity = total_capacity - total_resolved

        if total_capacity > 0:
            crew_utilization = total_resolved / total_capacity
        else:
            crew_utilization = 0.0

        distinct_zones = len(set(zone_list.tolist()))  # always num_crews now
        age_pressure = float(np.mean(np.log1p(self.backlog) * self.request_age))
        mean_request_age = float(np.mean(self.request_age))

        # --------------------------------------------------
        # 8. Reward
        #    distinct_zones bonus removed - structurally guaranteed now,
        #    so it would just be a constant offset, not a useful signal.
        # --------------------------------------------------
        reward = (
            0.1 * total_resolved
            - 0.1 * wasted_capacity
            - 0.05 * float(np.mean(self.backlog))
            - 0.4 * age_pressure
            - 0.2 * travel_cost
        )

        # --------------------------------------------------
        # 9. Advance simulation day
        # --------------------------------------------------
        self.current_day += 1
        terminated = self.current_day >= self.episode_length
        truncated = False

        if not terminated:
            self.forecast_window = self._load_forecast_window(
                self.start_date_idx + self.current_day
            )

        info = {
            "resolved": resolved,
            "new_requests": new_requests,
            "total_backlog": float(np.sum(self.backlog)),
            "total_resolved": total_resolved,
            "total_capacity": total_capacity,
            "wasted_capacity": wasted_capacity,
            "crew_utilization": crew_utilization,
            "travel_cost": float(travel_cost),
            "distinct_zones": distinct_zones,
            "mean_request_age": mean_request_age,
            "age_pressure": age_pressure,
        }

        return self._get_obs(), reward, terminated, truncated, info