import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import colorama
from tabulate import tabulate

colorama.init(autoreset=True)

df = pd.read_csv("Diabetes.csv")

X = df.drop('Diabetes', axis=1)
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

predicted = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predicted)
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
f1 = metrics.f1_score(y_test, predicted)
cm = metrics.confusion_matrix(y_test, predicted)

comparison = list(zip(y_test, predicted))

print("\n" + "="*50)
print(f"\033[1m{colorama.Fore.GREEN}DIABETES PREDICTION RESULTS{colorama.Fore.RESET}\033[0m")
print("="*50)

print("\n\033[1mACTUAL VS PREDICTED CLASSES:\033[0m")
headers = ["\033[94mActual\033[0m", "\033[92mPredicted\033[0m"]
print(tabulate(comparison[:10], headers=headers, tablefmt="simple"))
if len(comparison) > 10:
    print(f"... and {len(comparison)-10} more")

print("\n\033[1mCONFUSION MATRIX:\033[0m")
print(tabulate(cm, headers=['Predicted 0', 'Predicted 1'], 
               tablefmt="grid", showindex=['Actual 0', 'Actual 1']))

print("\n\033[1mCLASSIFICATION METRICS:\033[0m")
metrics_table = [
    ["Accuracy", f"{accuracy:.2%}"],
    ["Recall", f"{recall:.2%}"],
    ["Precision", f"{precision:.2%}"],
    ["F1-Score", f"{f1:.2%}"]
]
print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="simple"))

print("\n" + "="*50)