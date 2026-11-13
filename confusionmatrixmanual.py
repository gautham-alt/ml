import numpy as np

# Actual values
actual = np.array([1, 0, 1, 1, 0, 0, 1, 0])

# Predicted values
predicted = np.array([1, 0, 0, 1, 0, 1, 1, 0])

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(actual)):

    if actual[i] == 1 and predicted[i] == 1:
        TP += 1

    elif actual[i] == 0 and predicted[i] == 0:
        TN += 1

    elif actual[i] == 0 and predicted[i] == 1:
        FP += 1

    else:
        FN += 1

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

# Confusion matrix
cm = np.array([
    [TN, FP],
    [FN, TP]
])

print("Confusion Matrix:")
print(cm)
