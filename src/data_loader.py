'''

loading the data from kagglehub 
converting the xlsx file to csv file
then loading the file to datasets/raw 

'''


import pandas as pd
import os
import kagglehub

path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")

files = os.listdir(path)

excel_file = None

for file in files:
    if file.endswith('.xlsx'):
        excel_file = os.path.join(path, file)
        break


if excel_file is None:
    raise FileNotFoundError("No Excel file found in dataset folder.")

excel_path = os.path.join(path, excel_file)
df = pd.read_excel(excel_path)

csv_path = os.path.join("datasets/raw", "telco_customer_churn.csv")

df.to_csv(csv_path, index=False)
print(f"Data loaded and saved to {csv_path}")