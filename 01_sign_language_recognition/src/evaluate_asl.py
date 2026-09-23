import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATASET_PATH = "data/asl_alphabet"
MODEL_PATH = "outputs/asl_alphabet_model.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = val_ds.class_names

# Use same validation size as training
val_ds = val_ds.take(100)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating model...")

loss, accuracy = model.evaluate(val_ds, verbose=1)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

# Predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

os.makedirs("results", exist_ok=True)

with open("results/asl_classification_report.txt", "w") as f:
    f.write(report)

print("\nClassification Report:\n")
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("ASL Alphabet Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/asl_confusion_matrix.png",
    dpi=200
)

plt.close()

print("\nSaved:")
print("results/asl_classification_report.txt")
print("results/asl_confusion_matrix.png")
