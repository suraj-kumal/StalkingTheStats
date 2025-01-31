import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tabulate import tabulate
from colorama import Fore, Style, init

init(autoreset=True)
feature_names = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

wine_data = pd.read_csv(
    "D:/StalkingTheStats/wine/wine.data", header=None, names=feature_names
)

wine_data["target"] = wine_data["Class"]
X = wine_data.drop(["target", "Class"], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_scaled)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
wine_data["Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_principal[:, 0], X_principal[:, 1], c=wine_data["Cluster"], cmap="viridis"
)
plt.title("Wine Dataset - K-Means++ Clustering")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(scatter)
plt.show()

confusion_matrix = pd.crosstab(wine_data["target"], wine_data["Cluster"])

print(f"{Fore.RED}{Style.BRIGHT}Confusion Matrix:{Style.RESET_ALL}")
print(
    tabulate(
        confusion_matrix,
        headers=["Cluster 0", "Cluster 1", "Cluster 2"],
        tablefmt="pretty",
        showindex=True,
    )
)

