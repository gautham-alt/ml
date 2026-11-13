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

# Class labels
y = np.array([0, 0, 0, 1, 1, 1])

# Hyperplane parameters
w1 = 1
w2 = -1
b = 0

# Decision boundary
x1 = np.linspace(0, 10, 100)
x2 = -(w1 * x1 + b) / w2

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y)

# Plot decision boundary
plt.plot(x1, x2)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision Boundary")
plt.show()
