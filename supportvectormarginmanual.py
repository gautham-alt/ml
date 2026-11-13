import numpy as np
import matplotlib.pyplot as plt

# Data points
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6]
])

y = np.array([0, 0, 0, 1, 1, 1])

# Hyperplane parameters
w1 = 1
w2 = -1
b = 0

# Calculate distance of each point from hyperplane
distances = np.abs(w1 * X[:, 0] + w2 * X[:, 1] + b)
distances = distances / np.sqrt(w1**2 + w2**2)

# Find minimum distance
margin = np.min(distances)

print("Distances:", distances)
print("Margin:", margin)

# Plot points
plt.scatter(X[:, 0], X[:, 1], c=y)

# Decision boundary
x = np.linspace(0, 10, 100)
decision = x

plt.plot(x, decision, label="Decision Boundary")

plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
