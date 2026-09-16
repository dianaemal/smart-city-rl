import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.environment.data.zone_config import LOCAL_AREAS
from src.forecasting.prediction_loader import PredictionLoader
from src.environment.travel import TRAVEL_MATRIX


class CityServiceEnv(gym.Env):

    def __init__(self, date_indices=None, forecast_horizon=5):
        """
        date_indices: optional array of allowed start-date indices to sample
        from on reset(). Pass the TRAIN pool when training and the EVAL
        (held-out) pool when evaluating, so the two never overlap.
        If None, falls back to the full available range (old behavior).

        forecast_horizon: number of days of forecasted demand the agent can
        see ahead of today (today counts as day 0). This is the core
        "proactive" capability greedy structurally cannot use, since greedy
        only ever looks at the current observation.
        """

        # constants that don't change
        self.num_crews = 11
        self.num_zones = len(LOCAL_AREAS)
        self.crew_capacity = 30
        self.episode_length = 30
        self.forecast_horizon = forecast_horizon
        # variables
        self.current_day = 0
        self.forecast_window = np.zeros((self.forecast_horizon, self.num_zones))
        self.backlog = np.zeros(self.num_zones)
        self.crew_locations = np.zeros(
            self.num_crews,
            dtype=int
        )
        self.predictor = PredictionLoader()
        self.request_age = np.zeros(self.num_zones, dtype=np.float32)

        # restrict which start dates this env instance can sample
        self.date_indices = date_indices

        # the agent will observe the following:
        # forecasted demand per zone for the next `forecast_horizon` days
        # (today + future days), log1p(backlog) per zone, crew locations
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(
            self.num_zones * self.forecast_horizon
            + self.num_zones      # log backlog
            + self.num_zones      # request age
            + self.num_crews,
        ),
        dtype=np.float32
       
        )
        self.action_space = spaces.MultiDiscrete(
            [self.num_zones] * self.num_crews
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
        """
        Loads forecasted demand for `date_idx` (today) through
        `date_idx + forecast_horizon - 1` (today + future days).
        Clipped to the last available date if the horizon would run past
        the end of the dataset (shouldn't normally happen given the
        max_start adjustment below, but kept as a safety net).
        """
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
            low=0,
            high=self.num_zones,
            size=self.num_crews
        ).astype(np.int32)

        self.backlog = self.np_random.integers(
            low=0,
            high=10,
            size=self.num_zones
        ).astype(np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.array(action, dtype=np.int32)
        old_locations = self.crew_locations.copy()

        # --------------------------------------------------
        # 1. Agent assigns each crew to a zone for today
        # --------------------------------------------------
        self.crew_locations = action

        # --------------------------------------------------
        # 2. Calculate travel cost
        # --------------------------------------------------
        travel_cost = 0.0

        for crew_idx in range(self.num_crews):
            start_zone = old_locations[crew_idx]
            end_zone = action[crew_idx]
            travel_cost += TRAVEL_MATRIX[start_zone, end_zone]

        # --------------------------------------------------
        # 3. Today's actual requests arrive
        #    Generated around today's forecast.
        # --------------------------------------------------
        new_requests = self.np_random.normal(
            loc=self.forecast_window[0],
            scale=5
        )

        new_requests = np.round(new_requests)
        new_requests = np.maximum(0, new_requests).astype(np.float32)

        self.backlog += new_requests

        # Save backlog before service so we can update request age correctly
        backlog_before_service = self.backlog.copy()

        # --------------------------------------------------
        # 4. Compute service capacity in each zone
        # --------------------------------------------------
        service_capacity = np.zeros(
            self.num_zones,
            dtype=np.float32
        )

        for zone in action:
            service_capacity[zone] += self.crew_capacity

        # --------------------------------------------------
        # 5. Resolve requests
        # --------------------------------------------------
        resolved = np.minimum(
            self.backlog,
            service_capacity
        )

        self.backlog -= resolved

        # --------------------------------------------------
        # 6. Update request age
        #
        # Interpretation:
        # - Zones with remaining backlog get older.
        # - Fully cleared zones reset to age 0.
        # - Partially served zones have their age reduced based on
        #   how much of the backlog was resolved.
        # --------------------------------------------------
        self.request_age[self.backlog > 0] += 1
        self.request_age[self.backlog == 0] = 0

        served_fraction = np.zeros(
            self.num_zones,
            dtype=np.float32
        )

        nonzero_mask = backlog_before_service > 0

        served_fraction[nonzero_mask] = (
            resolved[nonzero_mask]
            / backlog_before_service[nonzero_mask]
        )

        self.request_age *= (1.0 - served_fraction)

        # --------------------------------------------------
        # 7. Compute metrics
        # --------------------------------------------------
        total_resolved = float(np.sum(resolved))
        total_capacity = float(np.sum(service_capacity))
        wasted_capacity = total_capacity - total_resolved

        if total_capacity > 0:
            crew_utilization = total_resolved / total_capacity
        else:
            crew_utilization = 0.0

        distinct_zones = len(set(action.tolist()))

        age_pressure = float(
            np.mean(self.backlog * self.request_age)
        )

        mean_request_age = float(
            np.mean(self.request_age)
        )

        # --------------------------------------------------
        # 8. Reward
        #
        # Good:
        # - resolve many requests
        # - cover multiple zones
        #
        # Bad:
        # - waste capacity
        # - leave large backlog
        # - leave old unresolved backlog
        # - travel too much
        # --------------------------------------------------
        reward = (
            0.1 * total_resolved
            - 0.1 * wasted_capacity
            - 0.05 * float(np.mean(self.backlog))
            - 0.01 * age_pressure
            + 0.8 * distinct_zones
            - 0.2 * travel_cost
        )

        # --------------------------------------------------
        # 9. Advance simulation day
        # --------------------------------------------------
        self.current_day += 1

        terminated = self.current_day >= self.episode_length
        truncated = False

        # --------------------------------------------------
        # 10. If episode continues, slide forecast window forward
        # --------------------------------------------------
        if not terminated:
            self.forecast_window = self._load_forecast_window(
                self.start_date_idx + self.current_day
            )

        # --------------------------------------------------
        # 11. Debugging/evaluation info
        # --------------------------------------------------
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