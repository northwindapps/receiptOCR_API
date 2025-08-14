import cv2
import pytesseract
import numpy as np

def preprocess_for_total(image_path):
    """
    Specific preprocessing for receipt totals like $17.18
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Aggressive shadow removal
    # Use morphological gradient to find text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine to enhance text
    enhanced = cv2.add(gray, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)
    
    # Step 2: Normalize lighting
    # Calculate local mean
    blur = cv2.GaussianBlur(enhanced, (51,51), 0)
    divided = cv2.divide(enhanced, blur, scale=255)
    
    # Step 3: Sharpen
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(divided, -1, kernel_sharpen)
    
    # Step 4: Threshold with multiple methods
    # Method 1: OTSU
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive
    adaptive = cv2.adaptiveThreshold(sharpened, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 10)
    
    # Step 5: Clean up
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned, adaptive, otsu

# Process the image
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\rotated.jpg'
cleaned, adaptive, otsu = preprocess_for_total(image_path)

# Save processed versions to debug
cv2.imwrite('cleaned.png', cleaned)
cv2.imwrite('adaptive.png', adaptive)
cv2.imwrite('otsu.png', otsu)

# Configure Tesseract for numbers and currency
custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789.$- '

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Try different versions
for img, name in [(cleaned, 'cleaned'), (adaptive, 'adaptive'), (otsu, 'otsu')]:
    text = pytesseract.image_to_string(img, lang='eng',config=custom_config)
    print(f"{name}: {text.strip()}")