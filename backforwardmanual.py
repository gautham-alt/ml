import numpy as np

# Input and expected output
X = np.array([[1, 0]])
Y = np.array([[1]])

# Initial weights and bias
W = np.array([[0.2], [0.3]])
b = 0.1

learning_rate = 0.1

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation
z = np.dot(X, W) + b
output = sigmoid(z)

print("Predicted Output:", output)

# Calculate error
error = Y - output

# Backpropagation
derivative = output * (1 - output)

delta = error * derivative

# Update weights and bias
W = W + learning_rate * np.dot(X.T, delta)
b = b + learning_rate * delta[0, 0]

print("Updated Weights:", W)
print("Updated Bias:", b)
