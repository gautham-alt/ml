import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 6])

# Polynomial coefficients
coeff = np.polyfit(x, y, 2)

# Polynomial equation
p = np.poly1d(coeff)

# Predicted values
y_pred = p(x)

plt.scatter(x, y)
plt.plot(x, y_pred)

plt.title("Polynomial Regression")
plt.show()
