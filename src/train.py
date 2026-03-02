import os
import joblib
from sklearn.model_selection import train_test_split
from src.pipeline import build_pipeline


MODEL_PATH = "models/gradient_boosting_telco.joblib"
FINAL_THRESHOLD = 0.40


def train_model(df, continuous_cols, categorical_cols, binary_numeric_cols):

    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pipeline = build_pipeline(
        continuous_cols,
        categorical_cols,
        binary_numeric_cols
    )

    pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": pipeline,
        "threshold": FINAL_THRESHOLD
    }, MODEL_PATH)

    return X_test, y_test