import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset
X = load_iris().data[:, :2]

# Initialize centroids
centroids = X[[0, 50]]

# K-Means
for i in range(10):

    distances = np.sqrt(
        np.sum((X[:, None] - centroids) ** 2, axis=2)
    )

    labels = np.argmin(distances, axis=1)

    new_centroids = np.array([
        X[labels == j].mean(axis=0)
        for j in range(2)
    ])

    if np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids

# Calculate inertia
inertia = 0

for i in range(len(X)):
    inertia += np.sum((X[i] - centroids[labels[i]]) ** 2)

print("Inertia:", inertia)

# Calculate Silhouette Score manually
silhouette = []

for i in range(len(X)):

    # Average distance within same cluster
    same = X[labels == labels[i]]

    a = np.mean([
        np.linalg.norm(X[i] - point)
        for point in same if not np.array_equal(X[i], point)
    ])

    # Average distance to other cluster
    other = X[labels != labels[i]]

    b = np.mean([
        np.linalg.norm(X[i] - point)
        for point in other
    ])

    s = (b - a) / max(a, b)

    silhouette.append(s)

score = np.mean(silhouette)

print("Silhouette Score:", score)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="X", s=200)

plt.title("K-Means Clustering")
plt.show()
