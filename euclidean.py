import numpy as np

# Two data points
p1 = np.array([2, 3])
p2 = np.array([5, 7])

# Calculate differences
diff = p1 - p2

# Square the differences
square = diff ** 2

# Sum the squared differences
total = np.sum(square)

# Calculate Euclidean distance
distance = np.sqrt(total)

print("Euclidean Distance:", distance)
