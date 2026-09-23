import tensorflow as tf
import numpy as np
import cv2
import os
import random

MODEL_PATH = "outputs/asl_alphabet_model.keras"
DATASET_PATH = "data/asl_alphabet"
OUTPUT_PATH = "results/asl_prediction_results.jpg"

IMG_SIZE = (128, 128)
NUM_IMAGES = 8

model = tf.keras.models.load_model(MODEL_PATH)

# Alphabet classes
class_names = sorted(os.listdir(DATASET_PATH))
class_names = [
    name for name in class_names
    if os.path.isdir(os.path.join(DATASET_PATH, name))
]

# Select random images
samples = []

for class_name in class_names:
    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if images:
        image_name = random.choice(images)
        samples.append((class_name, os.path.join(class_path, image_name)))

random.seed(42)
samples = random.sample(samples, NUM_IMAGES)

results = []

for actual_class, image_path in samples:

    # Read original 200x200 image
    image = cv2.imread(image_path)

    if image is None:
        continue

    original = image.copy()

    # Prepare image for model
    resized = cv2.resize(image, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_image = rgb.astype("float32")
    input_image = np.expand_dims(input_image, axis=0)

    # Prediction
    prediction = model.predict(input_image, verbose=0)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    correct = predicted_class.lower() == actual_class.lower()

    # Add border
    if correct:
        border_color = (0, 180, 0)
        status = "CORRECT"
    else:
        border_color = (0, 0, 220)
        status = "INCORRECT"

    original = cv2.copyMakeBorder(
        original,
        5, 5, 5, 5,
        cv2.BORDER_CONSTANT,
        value=border_color
    )

    # Add text
    cv2.putText(
        original,
        f"Actual: {actual_class.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        original,
        f"Predicted: {predicted_class.upper()}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        original,
        f"Confidence: {confidence:.1f}%",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        original,
        status,
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        border_color,
        2
    )

    results.append(original)

# Create 4x2 display
while len(results) < NUM_IMAGES:
    results.append(np.zeros_like(results[0]))

rows = []

for i in range(0, NUM_IMAGES, 2):
    row = np.hstack((results[i], results[i + 1]))
    rows.append(row)

output = np.vstack(rows)

os.makedirs("results", exist_ok=True)
cv2.imwrite(OUTPUT_PATH, output)

print("\nPrediction completed!")
print("Saved:")
print(OUTPUT_PATH)
