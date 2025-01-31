import pandas as pd
import numpy as np

dataset = pd.read_csv("employees.csv")
print("Original Dataset")
print(dataset.head(25))

dataset["Salary"] = dataset["Salary"].fillna(dataset["Salary"].mean())
print("filled missing data with mean")
print(dataset.head(25))

dataset = pd.read_csv("employees.csv")
print("Original Dataset")
print(dataset.head(25))

dataset["Salary"] = dataset["Salary"].interpolate(method="linear")
print("filled missing data with interpolation method")
print(dataset.head(25))
