import numpy as np

actual = np.array([3, 5, 7, 9])
predicted = np.array([2, 6, 8, 10])

# Calculate error
error = actual - predicted

# Square the error
squared_error = error ** 2

# Calculate mean squared error
mse = np.mean(squared_error)

print("Loss:", squared_error)
print("Cost (MSE):", mse)
