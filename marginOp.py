import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the receipt image
img = cv2.imread('test9.jpg', cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding for better text detection
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)

# Define a horizontal kernel for detecting text lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detect_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find contours of the lines
contours, _ = cv2.findContours(detect_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around each line for visualization
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 1)

# Display with matplotlib instead of cv2.imshow()
plt.figure(figsize=(10, 10))
plt.imshow(img_color)
plt.axis('off')
plt.show()

# Add margins and save each line
margin = 10  # pixels
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    x_new = max(x - margin, 0)
    y_new = max(y - margin, 0)
    w_new = min(w + 2*margin, img.shape[1] - x_new)
    h_new = min(h + 2*margin, img.shape[0] - y_new)

    # Crop the line with margins
    line_with_margin = img[y_new:y_new+h_new, x_new:x_new+w_new]
    cv2.imwrite(f'line_{y}.png', line_with_margin)
