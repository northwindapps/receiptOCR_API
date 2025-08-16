import cv2
import numpy as np
import pytesseract
import os

def preprocess_and_read_digits(img, scale_factor=5):
    # 1. Upscale (super important for tiny numbers)
    upscaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE to boost local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 4. Sharpen with unsharp masking (small sigma for thin strokes)
    blurred = cv2.GaussianBlur(enhanced, (0,0), sigmaX=1)
    sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)

    # 5. Adaptive thresholding (better than Otsu for uneven receipts)
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 5
    )

    # 6. Run Tesseract optimized for small text (digits)
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(binary, config=config).strip()

    return text, binary

# --- Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create folder to save crops
save_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\crops"
os.makedirs(save_dir, exist_ok=True)

# Example usage:
image_path = r"C:\Users\ABC\Documents\receiptYOLOProject\test30.jpg"
img = cv2.imread(image_path)

digits, preview = preprocess_and_read_digits(img)

print("OCR result:", digits)

# Save the cropped image
crop_filename = os.path.join(save_dir, f"preprocessed.png")                            
cv2.imwrite(crop_filename, preview)
print(f"Saved crop: {crop_filename}")

# Optional: show preprocessed image
cv2.imshow("Preprocessed", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()

