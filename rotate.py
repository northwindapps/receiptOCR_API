import cv2
import numpy as np

# Read image
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\chunk_crop_0_10.png'
image = cv2.imread(image_path)

# Get image dimensions
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# Create rotation matrix (positive angle = counter-clockwise)
M = cv2.getRotationMatrix2D(center, -0.5, 1.0)  # angle=2 degrees, scale=1.0

# Apply rotation
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

# Save or show
cv2.imwrite("rotated.jpg", rotated)
