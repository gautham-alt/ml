import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create dataset
X = np.array([
    [2, 3], [3, 4], [4, 3],
    [8, 8], [9, 10], [10, 9]
])

# Create KMeans model
model = KMeans(n_clusters=2, random_state=42, n_init=10)

# Train model and predict clusters
labels = model.fit_predict(X)

# Get centroids
centroids = model.cluster_centers_

# Display cluster assignments
print("Cluster Assignments:")
print(labels)

# Display centroids
print("Centroids:")
print(centroids)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=labels)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1],
            color="red", marker="X", s=200,
            label="Centroids")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering")

plt.legend()
plt.show()
