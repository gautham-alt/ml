import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load Iris dataset
iris = load_iris()

# Select first two features
X = iris.data[:100, :2]
y = iris.target[:100]

# Create SVM model
model = SVC(kernel="linear")

# Train the model
model.fit(X, y)

# Create mesh grid
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1

y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

# Predict values for mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.title("SVM Decision Boundary")

plt.show()
