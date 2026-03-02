import os
import joblib
import json
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


MODEL_PATH = "models/gradient_boosting_telco.joblib"
REPORT_PATH = "reports/evaluation_report.json"


def evaluate(X_test, y_test):

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "threshold": threshold
    }

    os.makedirs("reports", exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete. Report saved to:", REPORT_PATH)

    return metrics