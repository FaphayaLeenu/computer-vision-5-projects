import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

os.makedirs("outputs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names

print("\nClasses:", class_names)

# Improve input pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# --------------------------------------------------
# MOBILE NET V2 BASE MODEL
# --------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------

inputs = tf.keras.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model.save("outputs/cats_dogs_model.keras")

print("\nModel saved to:")
print("outputs/cats_dogs_model.keras")

# --------------------------------------------------
# TRAINING GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Cats vs Dogs Classification Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/training_accuracy.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# LOSS GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Cats vs Dogs Classification Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/training_loss.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

y_true = []
y_pred = []

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = (predictions > 0.5).astype(int).flatten()

    y_pred.extend(predictions)
    y_true.extend(labels.numpy().astype(int).flatten())

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nClassification Report:\n")
print(report)

with open(
    "results/classification_report.txt",
    "w"
) as f:
    f.write(report)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(
    ax=ax,
    cmap="Blues"
)

plt.title("Cats vs Dogs Confusion Matrix")

plt.savefig(
    "results/confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nEvaluation complete.")
