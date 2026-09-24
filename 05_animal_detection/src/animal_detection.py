from ultralytics import YOLO
import cv2
import os

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run animal detection
results = model("input.png", conf=0.4, verbose=False)

# Get first result
result = results[0]

# Save annotated result
os.makedirs("results", exist_ok=True)

annotated = result.plot()
cv2.imwrite("results/animal_detection_result.jpg", annotated)

# Display detected objects
print("\nDetected objects:")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    name = model.names[class_id]

    print(f"{name}: {confidence:.2f}")

print("\nResult saved to:")
print("results/animal_detection_result.jpg")
