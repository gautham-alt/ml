import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [2, 3],
    [3, 3],
    [3, 4],
    [8, 8],
    [9, 8],
    [8, 9]
])

# Initial centroids
centroids = X[[0, 3]]

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

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="X", s=200)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Clustering Plot")

plt.show()
