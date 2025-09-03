import cv2
import pytesseract

# --- Image path ---
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test11double.jpg'
# image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test7.jpg'
# image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test54.jpg'
original_image = cv2.imread(image_path)

# Parameters to tweak sharpness
strength = 1.5  # How much to boost edges (1.0 = no change)
blur_size = (0, 0)  # Let OpenCV figure size from sigma
sigma = 3  # How much to blur before subtracting

# Unsharp masking
blurred = cv2.GaussianBlur(original_image, blur_size, sigma)
sharpened = cv2.addWeighted(original_image, 1 + strength, blurred, -strength, 0)
original_image = sharpened

# --- Generate image variants ---
images = {
    "original": original_image,
    "brighter": cv2.convertScaleAbs(original_image, alpha=1.5, beta=50),  # slightly brighter
    "brighter++": cv2.convertScaleAbs(original_image, alpha=1.5, beta=150),  # slightly brighter
    "darker": cv2.convertScaleAbs(original_image, alpha=1.5, beta=-50),    # slightly darker
    # "darker--": cv2.convertScaleAbs(original_image, alpha=1.5, beta=-100)    # slightly darker
}

# --- Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Function to process one image variant ---
def process_image(image, label):
    print(f"\n=== Processing {label} image ===")
    layout_crop = image
    gray = cv2.cvtColor(layout_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresh, lang='eng').strip()
    data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data['conf'] if int(c) >= 0]
    avg_conf = sum(confidences) / max(1, len(confidences)) / 100

    if text and avg_conf > 0.5:
        print(f"OCR Confidence: {avg_conf:.2f}")
        print(f"Chunk Text by Pytesseract: {text}")
        print("=" * 40)
    for i, word in enumerate(data['text']):
        if word.strip().upper() == "TOTAL" or word.strip().upper() == "Total":
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            print(f"Found TOTAL at: x={x}, y={y}, w={w}, h={h}, conf={data['conf'][i]}")
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.imwrite("receipt_with_total.jpg", image)
    


# --- Run all variants ---
for label, img in images.items():
    process_image(img, label)
