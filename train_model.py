import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from preprocess import load_data, build_preprocessor, prepare_data


def clean_feature_name(name):

    name = name.replace("num__", "")
    name = name.replace("cat__", "")
    name = name.replace("_", " ")
    name = name.title()

    return name


def train():

    df = load_data("data/fraud_data.csv")

    X_train, X_test, y_train, y_test = prepare_data(df)

    preprocessor = build_preprocessor()

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # =============================
    # Metrics
    # =============================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }

    # =============================
    # Feature importance
    # =============================

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    clean_names = [clean_feature_name(f) for f in feature_names]

    feature_importance = dict(zip(clean_names, importances))

    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(feature_importance, "models/feature_importance.pkl")
    joblib.dump(metrics, "models/model_metrics.pkl")

    print("Model saved with metrics + feature importance")


if __name__ == "__main__":
    train()