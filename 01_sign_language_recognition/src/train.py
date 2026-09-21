import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# -----------------------------
# 1. Load the dataset
# -----------------------------

train_data = pd.read_csv("data/sign_mnist_train.csv")
test_data = pd.read_csv("data/sign_mnist_test.csv")

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)


# -----------------------------
# 2. Separate labels and images
# -----------------------------

X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values


# -----------------------------
# 3. Normalize pixel values
# -----------------------------

X_train = X_train / 255.0
X_test = X_test / 255.0


# -----------------------------
# 4. Reshape images
# -----------------------------

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# -----------------------------
# 5. Convert labels
# -----------------------------

# Dataset has labels 0-24, but 9 is missing.
# We convert them to 0-23 for the neural network.

label_values = sorted(np.unique(y_train))

label_to_index = {
    label: index for index, label in enumerate(label_values)
}

y_train = np.array([label_to_index[label] for label in y_train])
y_test = np.array([label_to_index[label] for label in y_test])

y_train = to_categorical(y_train, num_classes=24)
y_test = to_categorical(y_test, num_classes=24)


# -----------------------------
# 6. Create CNN model
# -----------------------------

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(24, activation="softmax")
])


# -----------------------------
# 7. Compile the model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 8. Train the model
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64
)


# -----------------------------
# 9. Evaluate the model
# -----------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


# -----------------------------
# 10. Save the model
# -----------------------------

model.save("outputs/sign_language_model.keras")

print("Model saved successfully!")


# -----------------------------
# 11. Save training graph
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("results/training_accuracy.png")
plt.close()

print("Training graph saved successfully!")