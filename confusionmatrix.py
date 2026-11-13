import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Actual values
actual = [1, 0, 1, 1, 0, 0, 1, 0]

# Predicted values
predicted = [1, 0, 0, 1, 0, 1, 1, 0]

# Calculate confusion matrix
cm = confusion_matrix(actual, predicted)

# Display confusion matrix
print("Confusion Matrix:")
print(cm)

# Visualize confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot()

plt.title("Confusion Matrix")
plt.show()
