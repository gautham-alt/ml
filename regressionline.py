import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 6])

# Calculate slope
m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)

# Calculate intercept
c = np.mean(y) - m * np.mean(x)

# Predicted values
y_pred = m * x + c

plt.scatter(x, y)
plt.plot(x, y_pred)

plt.title("Linear Regression")
plt.show()

print("Slope:", m)
print("Intercept:", c)
