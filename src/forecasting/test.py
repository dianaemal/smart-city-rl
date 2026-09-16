from src.forecasting.demand_predictor import DemandPredictor

predictor = DemandPredictor()


date = predictor.test_dates[0]
preds = predictor.predict_for_date(date)

print(date)
print(preds)
print(preds.shape)