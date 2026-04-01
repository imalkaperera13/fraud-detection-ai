import joblib
import pandas as pd


def load_model():
    return joblib.load("models/fraud_model.pkl")


def load_feature_importance():
    return joblib.load("models/feature_importance.pkl")


def load_metrics():
    return joblib.load("models/model_metrics.pkl")


def predict_transaction(model, amount, time, location, device, merchant):

    input_data = pd.DataFrame([{
        "amount": amount,
        "time": time,
        "location": location,
        "device": device,
        "merchant": merchant
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if probability < 0.3:
        risk = "Low"
    elif probability < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return prediction, probability, risk