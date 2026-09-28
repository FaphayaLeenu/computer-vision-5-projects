import os
import glob
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMG_SIZE = (160, 160)
MODEL_PATH = "outputs/cats_dogs_model.keras"

os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# --------------------------------------------------
# FIND MOBILENETV2 INSIDE THE MODEL
# --------------------------------------------------

def find_model_with_conv(model):

    for layer in model.layers:

        # Check whether this layer contains convolutional layers
        if isinstance(layer, tf.keras.Model):

            for sub_layer in layer.layers:

                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer

                # Check deeper nested models
                if isinstance(sub_layer, tf.keras.Model):

                    result = find_model_with_conv(sub_layer)

                    if result is not None:
                        return result

    return None


base_model = find_model_with_conv(model)

if base_model is None:
    raise ValueError("Could not find MobileNetV2 inside the model.")

print("Feature model:", base_model.name)

# --------------------------------------------------
# FIND LAST CONVOLUTIONAL LAYER
# --------------------------------------------------

conv_layer = None

for layer in reversed(base_model.layers):

    if isinstance(layer, tf.keras.layers.Conv2D):
        conv_layer = layer
        break

if conv_layer is None:
    raise ValueError("No convolutional layer found.")

print("Grad-CAM layer:", conv_layer.name)

# --------------------------------------------------
# FIND CLASSIFIER HEAD
# --------------------------------------------------

gap_layer = None
dropout_layer = None
dense_layer = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.GlobalAveragePooling2D
    ):
        gap_layer = layer

    elif isinstance(
        layer,
        tf.keras.layers.Dropout
    ):
        dropout_layer = layer

    elif isinstance(
        layer,
        tf.keras.layers.Dense
    ):
        dense_layer = layer

if gap_layer is None:
    raise ValueError("GlobalAveragePooling2D layer not found.")

if dense_layer is None:
    raise ValueError("Dense classification layer not found.")

# --------------------------------------------------
# VALIDATION IMAGE
# --------------------------------------------------

image_paths = (
    glob.glob("dataset/val/cat/*") +
    glob.glob("dataset/val/dog/*")
)

if len(image_paths) == 0:
    raise FileNotFoundError(
        "No validation images found."
    )

image_path = image_paths[0]

print("Using image:")
print(image_path)

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

original = cv2.imread(image_path)

if original is None:
    raise ValueError(
        "Could not read the selected image."
    )

original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

image = cv2.resize(
    original_rgb,
    IMG_SIZE
)

input_image = np.expand_dims(
    image.astype(np.float32),
    axis=0
)

# --------------------------------------------------
# MOBILE NET PREPROCESSING
# --------------------------------------------------

preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
    input_image
)

# --------------------------------------------------
# GRAD-CAM FEATURE MODEL
# --------------------------------------------------

feature_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=conv_layer.output
)

# --------------------------------------------------
# CALCULATE GRADIENTS
# --------------------------------------------------

with tf.GradientTape() as tape:

    conv_outputs = feature_model(
        preprocessed,
        training=False
    )

    tape.watch(conv_outputs)

    # Classification head
    x = gap_layer(conv_outputs)

    if dropout_layer is not None:
        x = dropout_layer(
            x,
            training=False
        )

    prediction = dense_layer(x)

    probability = prediction[:, 0]

# --------------------------------------------------
# GRADIENTS
# --------------------------------------------------

gradients = tape.gradient(
    probability,
    conv_outputs
)

if gradients is None:
    raise ValueError(
        "Gradients could not be calculated."
    )

# Average gradients
pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(1, 2)
)

conv_outputs = conv_outputs[0]

pooled_gradients = pooled_gradients[0]

# Weighted feature maps
heatmap = tf.reduce_sum(
    conv_outputs *
    pooled_gradients,
    axis=-1
)

# ReLU
heatmap = tf.maximum(
    heatmap,
    0
)

# Normalize
max_value = tf.reduce_max(heatmap)

if max_value > 0:
    heatmap /= max_value

heatmap = heatmap.numpy()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

prediction_value = float(
    prediction[0][0]
)

if prediction_value >= 0.5:

    predicted_class = "dog"
    confidence = prediction_value

else:

    predicted_class = "cat"
    confidence = 1 - prediction_value

print()
print("Prediction:", predicted_class)
print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)

# --------------------------------------------------
# CREATE HEATMAP
# --------------------------------------------------

heatmap_resized = cv2.resize(
    heatmap,
    (
        original_rgb.shape[1],
        original_rgb.shape[0]
    )
)

heatmap_uint8 = np.uint8(
    255 * heatmap_resized
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)

# --------------------------------------------------
# OVERLAY
# --------------------------------------------------

overlay = (
    0.6 * original_rgb +
    0.4 * heatmap_color
)

overlay = np.uint8(
    np.clip(
        overlay,
        0,
        255
    )
)

# --------------------------------------------------
# SAVE RESULT
# --------------------------------------------------

output_path = (
    "results/gradcam_cats_dogs.png"
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(original_rgb)

plt.title(
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence * 100:.2f}%"
)

plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(overlay)

plt.title(
    "Grad-CAM Explanation"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print("Grad-CAM result saved to:")
print(output_path)
