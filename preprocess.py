import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str):
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    return X, y


def build_preprocessor():
    numeric_features = ["amount", "time"]
    categorical_features = ["location", "device", "merchant"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor


def prepare_data(df: pd.DataFrame):
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test