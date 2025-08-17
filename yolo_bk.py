import cv2
import pytesseract
from ultralytics import YOLO

# --- Load models ---
layout_model = YOLO('layout_best.pt')         
chunk_model = YOLO('text_chunk_epoch40_best.pt')

# --- Image path ---
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test3.jpg'
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
    "darker--": cv2.convertScaleAbs(original_image, alpha=1.5, beta=-100)    # slightly darker
}

# --- Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Function to process one image variant ---
def process_image(image, label):
    print(f"\n=== Processing {label} image ===")
    layout_results = layout_model(source=image, conf=0.50, save=False, show=False)

    for layout_result in layout_results:
        for layout_box in layout_result.boxes:
            x_min, y_min, x_max, y_max = map(int, layout_box.xyxy[0].tolist())
            layout_crop = image[y_min:y_max, x_min:x_max]

            chunk_results = chunk_model(layout_crop, conf=0.3)

            for chunk_result in chunk_results:
                for chunk_box in chunk_result.boxes:
                    cx_min, cy_min, cx_max, cy_max = map(int, chunk_box.xyxy[0].tolist())
                    abs_x_min = x_min + cx_min
                    abs_y_min = y_min + cy_min
                    abs_x_max = x_min + cx_max
                    abs_y_max = y_min + cy_max

                    chunk_crop = layout_crop[cy_min:cy_max, cx_min:cx_max]
                    gray = cv2.cvtColor(chunk_crop, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                    text = pytesseract.image_to_string(thresh, lang='eng').strip()
                    data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
                    confidences = [int(c) for c in data['conf'] if int(c) >= 0]
                    avg_conf = sum(confidences) / max(1, len(confidences)) / 100

                    if text and avg_conf > 0.6:
                        print(f"Layout Class: {int(layout_box.cls[0])}, Conf: {float(layout_box.conf[0]):.2f}")
                        print(f"OCR Confidence: {avg_conf:.2f}")
                        print(f"Chunk Text: {text}")
                        print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")
                        print("=" * 40)

# --- Run all variants ---
for label, img in images.items():
    process_image(img, label)
