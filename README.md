# 📊 IBM Customer Churn Prediction — End-to-End ML Pipeline

An end-to-end Machine Learning pipeline built on the IBM Telco Customer Churn dataset.

This project follows a production-style ML workflow including:
- Data Engineering
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Comparison
- Evaluation & Metrics Analysis
- Experiment Tracking
- Reproducible Pipeline Design

---

## 🎯 Objective

Predict whether a telecom customer will churn based on demographic and service-related features.

Business Goal:
Reduce churn by identifying high-risk customers early.

---

## 📂 Project Structure

Project layout (matching the attached diagram):

ibm-churn-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md

---

## 🧠 ML Workflow

### 1️⃣ Data Engineering
- Handling missing values
- Data type corrections
- Outlier checks
- Target distribution analysis

### 2️⃣ EDA
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Business-driven insights

### 3️⃣ Feature Engineering
- Encoding categorical variables
- Feature scaling
- Feature selection
- Pipeline integration

### 4️⃣ Model Training
Models tested:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (optional)

### 5️⃣ Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## 📈 Results (To Be Updated)

Best Performing Model: _TBD_

ROC-AUC: _TBD_

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ibm-churn-ml.git
cd ibm-churn-ml