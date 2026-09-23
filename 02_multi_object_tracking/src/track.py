from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input video
input_video = "input.mp4"

# Output folder
os.makedirs("results", exist_ok=True)

# Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
output_path = "results/tracked_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Starting multi-object tracking...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection + ByteTrack tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False
    )

    # Draw tracking results
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    # Display
    cv2.imshow("Multi-Object Tracking", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Tracking completed!")
print(f"Output saved to: {output_path}")
