import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_data(df):
    try:
        df = df.copy()
        df["Temperatura Media (C)"] = df["Temperatura Media (C)"].apply(
            lambda x: float(str(x).replace(",", ".")) if isinstance(x, str) else x
        )
        df = df.dropna(subset=["Temperatura Media (C)", "Consumo de cerveja (litros)"])
        df["Temperatura Media (C)"] = pd.to_numeric(
            df["Temperatura Media (C)"], errors="coerce"
        )
        df["Consumo de cerveja (litros)"] = pd.to_numeric(
            df["Consumo de cerveja (litros)"], errors="coerce"
        )
        return df
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        raise


def calculate_coefficients(X, y):
    try:
        n = len(X)
        if n == 0:
            raise ValueError("Empty input arrays")
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        if denominator == 0:
            raise ValueError("Division by zero in coefficient calculation")
        beta1 = numerator / denominator
        beta0 = y_mean - (beta1 * X_mean)
        return beta0, beta1
    except Exception as e:
        print(f"Error in calculate_coefficients: {e}")
        raise


def predict(X, beta0, beta1):
    try:
        return beta0 + beta1 * X
    except Exception as e:
        print(f"Error in predict: {e}")
        raise


def calculate_r_squared(y_true, y_pred):
    try:
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        if ss_tot == 0:
            raise ValueError("Total sum of squares is zero")
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    except Exception as e:
        print(f"Error in calculate_r_squared: {e}")
        raise


def plot_regression_line(X, y, beta0, beta1):
    try:
        plt.scatter(X, y, color="blue", label="Actual Data")
        y_pred = predict(X, beta0, beta1)
        plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")
        plt.xlabel("Temperatura Media (C)")
        plt.ylabel("Consumo de cerveja (litros)")
        plt.title("Beer Consumption vs Temperature")
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error in plot_regression_line: {e}")
        raise


def run_linear_regression(file_path):
    try:
        df = pd.read_csv(file_path, decimal=",")
        df = prepare_data(df)
        X = df["Temperatura Media (C)"].values
        y = df["Consumo de cerveja (litros)"].values
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid data points after preparation")
        print(f"Number of valid data points: {len(X)}")
        beta0, beta1 = calculate_coefficients(X, y)
        y_pred = predict(X, beta0, beta1)
        r_squared = calculate_r_squared(y, y_pred)
        plot_regression_line(X, y, beta0, beta1)
        return {
            "intercept": beta0,
            "coefficient": beta1,
            "r_squared": r_squared,
            "equation": f"y = {beta0:.2f} + {beta1:.2f}x",
            "n_points": len(X),
        }
    except Exception as e:
        print(f"Error in run_linear_regression: {e}")
        return None


results = run_linear_regression("Consumo_cerveja.csv")
if results:
    print("\nLinear Regression Results:")
    print(f"Number of data points: {results['n_points']}")
    print(f"Equation: {results['equation']}")
    print(f"R-squared: {results['r_squared']:.4f}")
    new_temp = 32.0
    prediction = results["intercept"] + results["coefficient"] * new_temp
    print(f"\nPredicted beer consumption for {new_temp}°C: {prediction:.2f} liters")
else:
    print("\nRegression analysis failed. Please check the error messages above.")
