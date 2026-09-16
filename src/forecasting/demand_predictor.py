import joblib
import pandas as pd
import numpy as np
from src.environment.data.zone_config import LOCAL_AREAS
from src.environment.data.zone_config import ZONE_TO_ID

class DemandPredictor:

    def __init__(
            self,
            model_path= "results/models/xgboost_demand_model.pkl",
            feature_path = "results/models/xgboost_feature_columns.pkl",
            data_path = "data/forecasting_dataset.csv"
    ):
        self.model = joblib.load(model_path)
        self.features = joblib.load(feature_path)
        self.df = pd.read_csv(data_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["area_encoded"] = self.df["Local area"].map(ZONE_TO_ID)

        # make a sorted unique dates list
        self.available_dates = sorted(self.df["date"].unique())

        self.test_dates = [
            d
            for d in self.available_dates
            if d >= pd.Timestamp("2026-01-02")
        ]


    def predict_for_date(self, date):
        date = pd.to_datetime(date)

        day_df = self.df[self.df["date"]== date].copy()

        # we must put rows in the same order as our environment accepts it (A - Z)
        day_df = (
            day_df
            .set_index("Local area")
            .reindex(LOCAL_AREAS)
            .reset_index()
        )

        X = day_df[self.features]

        preds = self.model.predict(X)
        preds = np.maximum(0, np.round(preds)).astype(np.float32)

        return preds



