import numpy as np

# Actual values
actual = np.array([3, 5, 7, 9, 11])

# Predicted values
predicted = np.array([2, 6, 8, 10, 10])

# Calculate mean
mean = np.mean(actual)

# Residual sum of squares
ss_res = np.sum((actual - predicted) ** 2)

# Total sum of squares
ss_tot = np.sum((actual - mean) ** 2)

# Calculate R-squared
r2 = 1 - (ss_res / ss_tot)

print("R-squared:", r2)
