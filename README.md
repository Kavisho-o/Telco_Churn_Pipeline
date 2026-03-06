# 📊 Telco Customer Churn Prediction — End-to-End ML Pipeline

An end-to-end reproducible pipeline that predicts customer churn using the IBM Telco dataset. The final Gradient Boosting model achieves the following on the test set:

- ROC-AUC: 0.85498
- Precision: 0.6069
- Recall: 0.6604
- F1: 0.6325

The trained model artifact is saved as `models/gradient_boosting_telco.joblib` and the evaluation metrics are in `reports/evaluation_report.json`.

## Why this project

- Clear problem framing and business objective (reduce customer churn).
- End-to-end reproducible pipeline: data ingestion, feature engineering, model training, evaluation.
- Good model-performance and evaluation using business-relevant metrics (ROC-AUC, precision/recall tradeoffs).
- Clean repo structure, runnable entrypoint, and explanatory notebooks demonstrating EDA and feature decisions.

## Quick Project Summary

- Problem: Binary classification — will a customer churn?
- Dataset: IBM Telco Customer Churn (included under `datasets/raw/Telco-Customer-Churn.csv`).
- Approach: EDA → feature engineering → model training & tuning → evaluation and saved pipeline.

## Repo structure (relevant files)

```
.
├── main.py                          # runnable pipeline entrypoint
├── datasets/
│   ├── raw/Telco-Customer-Churn.csv
│   └── processed/feature_engineered_telco.csv
├── notebooks/                       # EDA, feature engineering, model training exploration
├── src/
│   ├── data_loader.py               # load and validate raw data
│   ├── features.py                  # feature engineering functions
│   ├── pipeline.py                  # pipeline utilities
│   ├── train.py                     # training loop (returns X_test, y_test)
│   └── evaluate.py                  # evaluation & metric reporting
├── models/
│   └── gradient_boosting_telco.joblib
├── reports/
│   └── evaluation_report.json
├── requirements.txt
└── README.md
```

## What I did 

- Data cleaning and preprocessing (see `src/data_loader.py`).
- Feature engineering including categorical encodings and derived buckets (see `src/features.py`).
- Built and compared multiple models; selected a Gradient Boosting model saved under `models/`.
- Evaluated with ROC-AUC and class-specific metrics; report saved in `reports/evaluation_report.json`.

## Reproduce locally

1. Create and activate a Python virtual environment (Windows):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the full pipeline (loads raw data, applies features, trains model, evaluates):

```powershell
python main.py
```

3. Inspect results:

- Model artifact: `models/gradient_boosting_telco.joblib`
- Evaluation metrics: `reports/evaluation_report.json`
- Notebooks for the exploratory work: `notebooks/eda.ipynb`, `notebooks/feature_engg.ipynb`, `notebooks/model_training.ipynb`

## How recruiters can validate my work quickly

- Open `main.py` and run `python main.py` to reproduce training and evaluation.
- Review `notebooks/` for my EDA, feature decisions and experiments.
- Check `src/` for cleanly organized, callable functions that support automated runs.

## Notes & Tips

- Threshold tuning: the evaluation report uses a decision threshold of 0.4 (see `reports/evaluation_report.json`) to balance precision/recall for the business context.
- The pipeline is intentionally modular to make it straightforward to add new models, metrics, or input features.

## Contact

If you'd like to see this deployed as a demo or want the runnable environment, contact: kavisheleven011@gmail.com

---
