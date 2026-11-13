from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Actual values
actual = [1, 0, 1, 1, 0, 0, 1, 0]

# Predicted values
predicted = [1, 0, 0, 1, 0, 1, 1, 0]

# Calculate evaluation metrics
accuracy = accuracy_score(actual, predicted)

precision = precision_score(actual, predicted)

recall = recall_score(actual, predicted)

f1 = f1_score(actual, predicted)

# Display results
print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)
