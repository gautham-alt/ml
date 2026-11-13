import numpy as np

# Actual values
actual = np.array([3, 5, 7, 9, 11])

# Predicted values
predicted = np.array([2, 6, 8, 10, 10])

# Calculate error
error = actual - predicted

# Square the error
squared_error = error ** 2

# Calculate MSE
mse = np.sum(squared_error) / len(actual)

print("MSE:", mse)
