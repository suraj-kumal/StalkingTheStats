import pandas as pd
import numpy as np

dataset = pd.read_csv("employees.csv")


# original Dataset
print("Original Dataset")
print(dataset.head(25))


dataset = dataset.dropna(axis=0)

dataset = dataset.drop_duplicates(keep="first")
dataset = dataset.drop(columns=["Bonus %"])

dataset["Team"] = dataset["Team"].replace(
    {"Fin": "Finance", "Mkt": "Marketing", "Financeance": "Finance"}, regex=True
)

print(dataset.head(25))
dataset.to_csv("employees_cleaned.csv", index=False)
print("Successfully Cleaning complete")
