import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = "Consumo_cerveja.csv" 
data = pd.read_csv(file_path)

columns_to_convert = [
    "Temperatura Media (C)",
    "Temperatura Minima (C)",
    "Temperatura Maxima (C)",
    "Precipitacao (mm)",
    "Consumo de cerveja (litros)",
]

for col in columns_to_convert:
    if data[col].dtype == "object":
        data[col] = data[col].str.replace(",", ".").astype(float)

data = data.dropna(subset=["Consumo de cerveja (litros)"])

data = data.drop(columns=["Data"])

X = data.drop(columns=["Consumo de cerveja (litros)"])
y = data["Consumo de cerveja (litros)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree_model = DecisionTreeRegressor(criterion="squared_error", random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

random_data = pd.DataFrame(
    {
        "Temperatura Media (C)": [25.0],
        "Temperatura Minima (C)": [22.0],
        "Temperatura Maxima (C)": [30.0],
        "Precipitacao (mm)": [0.5],
        "Final de Semana": [1],
    }
)

# Make a prediction
predicted_consumption = tree_model.predict(random_data)
print("Predicted Beer Consumption (liters):", predicted_consumption[0])
