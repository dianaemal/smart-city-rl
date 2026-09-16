import pandas as pd
import numpy as np

from src.environment.data.zone_config import LOCAL_AREAS


class PredictionLoader:
    def __init__(self, path="data/predicted_demand_2026.csv"):
        self.df = pd.read_csv(path)
        self.df["date"] = pd.to_datetime(self.df["date"])

        self.available_dates = sorted(self.df["date"].unique())

    def get_prediction_for_date(self, date):
        date = pd.to_datetime(date)

        day_df = self.df[self.df["date"] == date].copy()

        day_df = (
            day_df
            .set_index("Local area")
            .reindex(LOCAL_AREAS)
            .reset_index()
        )

        preds = day_df["predicted_demand"].to_numpy(dtype=np.float32)

        return preds