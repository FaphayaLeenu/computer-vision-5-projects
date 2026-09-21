import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# -----------------------------
# 1. Load test dataset
# -----------------------------

test_data = pd.read_csv("data/sign_mnist_test.csv")

X_test = test_data.drop("label", axis=1).values
y_test_original = test_data["label"].values


# -----------------------------
# 2. Prepare images
# -----------------------------

X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)


# -----------------------------
# 3. Convert labels
# -----------------------------

label_values = sorted(np.unique(y_test_original))

label_to_index = {
    label: index for index, label in enumerate(label_values)
}

y_test = np.array([
    label_to_index[label] for label in y_test_original
])


# -----------------------------
# 4. Load trained model
# -----------------------------

model = load_model("outputs/sign_language_model.keras")


# -----------------------------
# 5. Make predictions
# -----------------------------

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)


# -----------------------------
# 6. Calculate accuracy
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("Test Accuracy (%):", round(accuracy * 100, 2))


# -----------------------------
# 7. Classification report
# -----------------------------

letters = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]

report = classification_report(
    y_test,
    y_pred,
    target_names=letters
)

print("\nClassification Report:\n")
print(report)


# -----------------------------
# 8. Save classification report
# -----------------------------

with open("results/classification_report.txt", "w") as file:
    file.write("Vision-Based Sign Language Recognition\n")
    file.write("=======================================\n\n")
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)


# -----------------------------
# 9. Confusion matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=letters,
    yticklabels=letters
)

plt.title("Confusion Matrix - Sign Language Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig("results/confusion_matrix.png")
plt.close()


print("\nConfusion matrix saved!")
print("Classification report saved!")