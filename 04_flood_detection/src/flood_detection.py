import cv2
import numpy as np
import os

INPUT_IMAGE = "input.jpg"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read image
image = cv2.imread(INPUT_IMAGE)

if image is None:
    print("Error: Could not read input.jpg")
    exit()

original = image.copy()

height, width = image.shape[:2]

# ---------------------------------------
# Convert to HSV
# ---------------------------------------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ---------------------------------------
# Detect muddy/low-saturation water
# ---------------------------------------
lower = np.array([0, 0, 50])
upper = np.array([35, 125, 205])

color_mask = cv2.inRange(hsv, lower, upper)

# ---------------------------------------
# Create region of interest
# Focus on the river/flooded area
# ---------------------------------------
roi = np.zeros((height, width), dtype=np.uint8)

points = np.array([
    [0, int(height * 0.55)],
    [int(width * 0.25), int(height * 0.43)],
    [int(width * 0.55), int(height * 0.45)],
    [width, int(height * 0.58)],
    [width, int(height * 0.75)],
    [int(width * 0.70), int(height * 0.70)],
    [int(width * 0.40), int(height * 0.68)],
    [int(width * 0.15), int(height * 0.72)]
], dtype=np.int32)

cv2.fillPoly(roi, [points], 255)

# Combine color detection with ROI
mask = cv2.bitwise_and(color_mask, roi)

# ---------------------------------------
# Remove noise
# ---------------------------------------
kernel = np.ones((9, 9), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# ---------------------------------------
# Keep large connected regions
# ---------------------------------------
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    mask,
    connectivity=8
)

clean_mask = np.zeros_like(mask)

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area > 3000:
        clean_mask[labels == i] = 255

# ---------------------------------------
# Create flood overlay
# ---------------------------------------
overlay = original.copy()

flood_color = np.zeros_like(original)
flood_color[:, :] = (255, 0, 0)

overlay[clean_mask > 0] = flood_color[clean_mask > 0]

result = cv2.addWeighted(
    original,
    0.70,
    overlay,
    0.30,
    0
)

# ---------------------------------------
# Add label
# ---------------------------------------
cv2.putText(
    result,
    "Detected Flood / Water Region",
    (30, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

# ---------------------------------------
# Save outputs
# ---------------------------------------
cv2.imwrite(
    os.path.join(OUTPUT_DIR, "flood_mask.png"),
    clean_mask
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "flood_detection_result.jpg"),
    result
)

print("Flood detection completed!")
print("Flood mask saved to results/flood_mask.png")
print("Detection result saved to results/flood_detection_result.jpg")

# ---------------------------------------
# Display
# ---------------------------------------
cv2.imshow("Original Image", original)
cv2.imshow("Flood Detection", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
