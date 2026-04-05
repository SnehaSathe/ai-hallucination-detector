import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


from app.detector import detect_hallucination

# Load dataset
with open("data/dataset.json") as f:
    data = json.load(f)

y_true = []
y_pred = []

# Run predictions
for item in data:
    result = detect_hallucination(item["answer"], item["context"])

    y_true.append(item["label"])
    y_pred.append(result["label"])

# Metrics
print("\n📊 Evaluation Metrics:\n")

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='weighted'))
print("Recall:", recall_score(y_true, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))

# Confusion Matrix
print("\n📉 Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# Detailed Report
print("\n📋 Classification Report:\n")
print(classification_report(y_true, y_pred))


ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.show()