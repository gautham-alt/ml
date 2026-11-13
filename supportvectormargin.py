import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load Iris dataset
iris = load_iris()

# Select first 100 samples and two features
X = iris.data[:100, :2]
y = iris.target[:100]

# Create linear SVM model
model = SVC(kernel="linear", C=1)

# Train the model
model.fit(X, y)

# Display support vectors
print("Support Vectors:")
print(model.support_vectors_)

print("Number of Support Vectors:")
print(model.n_support_)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y)

# Highlight support vectors
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=150, facecolors="none",
            edgecolors="red", label="Support Vectors")

# Create decision boundary and margins
ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()

import numpy as np

xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)

YY, XX = np.meshgrid(yy, xx)

xy = np.c_[XX.ravel(), YY.ravel()]

Z = model.decision_function(xy).reshape(XX.shape)

plt.contour(XX, YY, Z, levels=[-1, 0, 1],
            colors="black",
            linestyles=["--", "-", "--"])

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("SVM Support Vectors and Margin")

plt.legend()
plt.show()
