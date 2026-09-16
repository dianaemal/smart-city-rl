import os
import joblib
import pandas as pd
import numpy as np

from src.environment.data.zone_config import LOCAL_AREAS, ZONE_TO_ID


MODEL_PATH = "results/models/xgboost_demand_model.pkl"
FEATURE_PATH = "results/models/xgboost_feature_columns.pkl"
DATA_PATH = "data/forecasting_dataset.csv"
OUTPUT_PATH = "data/predicted_demand_2026.csv"


def main():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    df = df[df["date"] >= "2026-01-01"].copy()

    df["area_encoded"] = df["Local area"].map(ZONE_TO_ID)

    all_predictions = []

    for date in sorted(df["date"].unique()):
        day_df = df[df["date"] == date].copy()

        day_df = (
            day_df
            .set_index("Local area")
            .reindex(LOCAL_AREAS)
            .reset_index()
        )

        day_df["area_encoded"] = day_df["Local area"].map(ZONE_TO_ID)

        if day_df[feature_columns].isna().any().any():
            continue

        X = day_df[feature_columns]

        preds = model.predict(X)
        preds = np.maximum(0, np.round(preds)).astype(int)

        day_df["predicted_demand"] = preds
        day_df["date"] = date

        all_predictions.append(
            day_df[["date", "Local area", "predicted_demand"]]
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    os.makedirs("data/processed", exist_ok=True)
    predictions_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print(predictions_df.head())
    print(predictions_df.shape)


if __name__ == "__main__":
    main()