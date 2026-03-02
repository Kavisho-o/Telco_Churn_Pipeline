from src.data_loader import load_raw_data
from src.features import feature_engineering
from src.train import train_model
from src.evaluate import evaluate


def main():

    print("Loading raw data...")
    df = load_raw_data()

    print("Applying feature engineering...")
    df = feature_engineering(df)

    # Define feature categories
    continuous_cols = ["tenure", "Monthly Charges", "Total Services"]
    categorical_cols = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Payment Method",
        "Tenure Bucket"
    ]
    binary_numeric_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Is Month-to-Month",
        "High Monthly Charges"
    ]

    print("Training model...")
    X_test, y_test = train_model(
        df,
        continuous_cols,
        categorical_cols,
        binary_numeric_cols
    )

    print("Evaluating model...")
    evaluate(X_test, y_test)

    print("Pipeline execution complete.")


if __name__ == "__main__":
    main()