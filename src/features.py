import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Performs cleaning and feature engineering 
    on the Telco Customer Churn dataset.

    '''

    # Dropping leakage and irrelevant features

    drop_cols = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
        "CLTV",
        "Total Charges",
    ]


    df = df.drop(columns=drop_cols)

    # Normalize tenure column name (some datasets use 'Tenure Months')
    if "tenure" not in df.columns:
        if "Tenure Months" in df.columns:
            df["tenure"] = df["Tenure Months"]
        elif "tenure_months" in df.columns:
            df["tenure"] = df["tenure_months"]

    # Ensure tenure is numeric
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)


    # Fixing internet dependent services

    internet_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies"
    ]


    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0})


    
    # Tenure Bucket (low, medium, high)

    def tenure_bucket(tenure):

        if tenure <= 12:
            return "low"
        
        elif tenure <= 36:
            return "medium"
        
        else:
            return "high"
    
    df["Tenure Bucket"] = df["tenure"].apply(tenure_bucket)



    # Total Services 


    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies"
    ]


    df["Total Services"] = df[service_cols].sum(axis=1)


    # Is Month-to-Month

    df["Is Month-to-Month"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    df = df.drop(columns=["Contract"])



    # High Monthly Charges

    threshold = df["Monthly Charges"].median()

    df["High Monthly Charges"] = (df["Monthly Charges"] > threshold).astype(int)



    return df