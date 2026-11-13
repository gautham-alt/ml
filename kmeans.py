import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([
    [2, 3],
    [3, 3],
    [3, 4],
    [8, 8],
    [9, 8],
    [8, 9]
])

# Initial centroids
centroids = np.array([
    [2, 3],
    [8, 8]
])

# Repeat clustering
for i in range(10):

    # Calculate distances
    distances = np.zeros((len(X), len(centroids)))

    for j in range(len(centroids)):
        distances[:, j] = np.sqrt(
            np.sum((X - centroids[j]) ** 2, axis=1)
        )

    # Assign clusters
    labels = np.argmin(distances, axis=1)

    # Calculate new centroids
    new_centroids = np.array([
        X[labels == j].mean(axis=0)
        for j in range(len(centroids))
    ])

    # Check convergence
    if np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids

print("Cluster Assignments:", labels)
print("Final Centroids:\n", centroids)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="X", s=200, label="Centroids")

plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.title("K-Means Clustering")
plt.show()
