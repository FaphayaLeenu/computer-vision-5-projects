import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------
# 1. Load trained model
# -----------------------------
model = load_model("outputs/sign_language_model.keras")

# -----------------------------
# 2. Load test dataset
# -----------------------------
df = pd.read_csv("data/sign_mnist_test.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

# Original Sign Language MNIST labels
class_labels = sorted(df["label"].unique())

# Convert pixels to images
X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# -----------------------------
# 3. Select 4 samples
# -----------------------------
indices = [100, 1000, 3000, 5000]

images = X[indices]
actual_labels = y[indices]

# -----------------------------
# 4. Predict
# -----------------------------
predictions = model.predict(images, verbose=0)

predicted_indices = np.argmax(predictions, axis=1)

# Convert model class index back to original dataset label
predicted_labels = [
    class_labels[index] for index in predicted_indices
]

# ASL alphabet mapping
letter_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y"
}

# -----------------------------
# 5. Create clean visualization
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

fig.suptitle(
    "Vision-Based Sign Language Recognition\nCNN Prediction Results",
    fontsize=20,
    fontweight="bold"
)

for i, ax in enumerate(axes.flat):

    ax.imshow(
        images[i].squeeze(),
        cmap="gray",
        interpolation="nearest"
    )

    ax.axis("off")

    actual = letter_map[actual_labels[i]]
    predicted = letter_map[predicted_labels[i]]

    if actual == predicted:
        result = "CORRECT"
    else:
        result = "INCORRECT"

    ax.set_title(
        f"Actual: {actual}    |    Predicted: {predicted}\n{result}",
        fontsize=15,
        fontweight="bold"
    )

plt.tight_layout(rect=[0, 0, 1, 0.93])

# -----------------------------
# 6. Save high-quality output
# -----------------------------
plt.savefig(
    "results/prediction_samples.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Clean prediction visualization saved successfully!")
