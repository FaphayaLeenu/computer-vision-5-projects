import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model
model = load_model("outputs/sign_language_model.keras")

# Load test dataset
df = pd.read_csv("data/sign_mnist_test.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

# Convert to images
X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Select 6 samples
indices = np.linspace(0, len(X) - 1, 6, dtype=int)

images = X[indices]
actual_labels = y[indices]

# Predict
predictions = model.predict(images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# Correct ASL label mapping
label_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N",
    14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V",
    22: "W", 23: "X", 24: "Y"
}

# Create larger figure
plt.figure(figsize=(15, 6))

for i in range(len(images)):

    plt.subplot(2, 3, i + 1)

    plt.imshow(
        images[i].squeeze(),
        cmap="gray",
        interpolation="nearest"
    )

    plt.axis("off")

    actual = label_map.get(actual_labels[i], "?")
    predicted = label_map.get(predicted_labels[i], "?")

    if actual == predicted:
        title_color = "green"
    else:
        title_color = "red"

    plt.title(
        f"Actual: {actual}   |   Predicted: {predicted}",
        fontsize=16,
        color=title_color,
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    "results/prediction_samples.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Improved prediction samples saved successfully!")