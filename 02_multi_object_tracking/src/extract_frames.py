import cv2
import os

video_path = "results/tracked_output.mp4"
output_dir = "screenshots"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open tracked video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Take screenshots from different parts of the video
positions = [
    int(total_frames * 0.20),
    int(total_frames * 0.50),
    int(total_frames * 0.80)
]

for i, frame_number in enumerate(positions, start=1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if ret:
        filename = os.path.join(
            output_dir,
            f"tracking_frame_{i}.jpg"
        )

        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

cap.release()

print("Screenshots created successfully!")
