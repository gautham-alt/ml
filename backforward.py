import numpy as np

# Input and target output
X = np.array([[1, 0]])
y = np.array([[1]])

# Initialize weights
W1 = np.array([[0.5, 0.2],
               [0.3, 0.4]])

W2 = np.array([[0.6],
               [0.7]])

learning_rate = 0.1

# Forward Propagation

# Hidden layer
hidden_input = np.dot(X, W1)
hidden_output = 1 / (1 + np.exp(-hidden_input))

# Output layer
final_input = np.dot(hidden_output, W2)
output = 1 / (1 + np.exp(-final_input))

print("Predicted Output:", output)

# Calculate error
error = y - output

print("Error:", error)

# Backpropagation

# Output layer gradient
d_output = error * output * (1 - output)

# Hidden layer gradient
d_hidden = np.dot(d_output, W2.T) * hidden_output * (1 - hidden_output)

# Update weights
W2 = W2 + learning_rate * np.dot(hidden_output.T, d_output)

W1 = W1 + learning_rate * np.dot(X.T, d_hidden)

print("Updated W1:")
print(W1)

print("Updated W2:")
print(W2)
