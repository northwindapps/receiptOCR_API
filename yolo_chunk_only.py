import cv2
import pytesseract
from ultralytics import YOLO

# Load chunk model only
chunk_model = YOLO('text_chunk_epoch40_best.pt')

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test7.jpg'
image = cv2.imread(image_path)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Run chunk detection on full image
chunk_results = chunk_model(image_path, conf=0.6, save=True, show=True)

for chunk_result in chunk_results:
    for chunk_box in chunk_result.boxes:
        # Chunk coords in original image
        x_min, y_min, x_max, y_max = map(int, chunk_box.xyxy[0].tolist())

        # Crop chunk from original image
        chunk_crop = image[y_min:y_max, x_min:x_max]

        # Preprocess for OCR
        gray = cv2.cvtColor(chunk_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # OCR
        text = pytesseract.image_to_string(thresh, lang='eng').strip()
        data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
        avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
        avg_conf /= 100
        if text != '' and avg_conf > 0.6:
            print(f"Average OCR Confidence: {avg_conf:.2f}")

            print(f"Chunk Text: {text}")
            print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            print("="*40)
