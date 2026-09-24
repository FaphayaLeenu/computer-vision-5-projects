import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------
# Settings
# -----------------------------
IMAGE_PATH = "input.jpg"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load pretrained MobileNetV2
# -----------------------------
model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)

# -----------------------------
# Load and preprocess image
# -----------------------------
img = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(
    img_array.copy()
)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(processed_img, verbose=0)

predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
    predictions,
    top=1
)[0][0]

label = decoded[1]
confidence = decoded[2]

print(f"Predicted object: {label}")
print(f"Confidence: {confidence * 100:.2f}%")

# -----------------------------
# Find last convolution layer
# -----------------------------
last_conv_layer = model.get_layer("Conv_1")

grad_model = tf.keras.models.Model(
    model.inputs,
    [last_conv_layer.output, model.output]
)

# -----------------------------
# Calculate Grad-CAM
# -----------------------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(processed_img)
    class_output = predictions[:, predicted_class]

grads = tape.gradient(
    class_output,
    conv_outputs
)

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.maximum(tf.reduce_max(heatmap), 1e-10)

heatmap = heatmap.numpy()

# -----------------------------
# Create heatmap
# -----------------------------
original = cv2.imread(IMAGE_PATH)

heatmap_resized = cv2.resize(
    heatmap,
    (original.shape[1], original.shape[0])
)

heatmap_uint8 = np.uint8(
    255 * heatmap_resized
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    original,
    0.6,
    heatmap_color,
    0.4,
    0
)

# -----------------------------
# Save result
# -----------------------------
output_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_result.jpg"
)

cv2.imwrite(
    output_path,
    overlay
)

# -----------------------------
# Display result
# -----------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {label}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Explanation")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "explainable_ai_result.png"),
    dpi=200
)

plt.show()

print("Grad-CAM result saved successfully!")
