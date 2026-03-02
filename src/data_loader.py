import pandas as pd
import os


RAW_PATH = "datasets/raw/Telco-Customer-Churn.csv"


def load_raw_data() -> pd.DataFrame:

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH)
    return df