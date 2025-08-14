import cv2
import os
import pytesseract
from ultralytics import YOLO

# Load chunk model only
chunk_model = YOLO('text_chunk_epoch40_best.pt')

# Create folder to save crops
save_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\crops"
os.makedirs(save_dir, exist_ok=True)

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test3.jpg'
image = cv2.imread(image_path)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Run chunk detection on full image
chunk_results = chunk_model(source=image, conf=0.6, save=True, show=True)

for idx,chunk_result in enumerate(chunk_results):
    for jdx, chunk_box in enumerate(chunk_result.boxes):
        # Chunk coords in original image
        x_min, y_min, x_max, y_max = map(int, chunk_box.xyxy[0].tolist())

        # Crop chunk from original image
        chunk_crop = image[y_min:y_max, x_min:x_max]

        bc_image = cv2.convertScaleAbs(src=chunk_crop, alpha=2.0, beta=-50)

        # Preprocess for OCR
        gray = cv2.cvtColor(bc_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Save the cropped image
        crop_filename = os.path.join(save_dir, f"chunk_crop_{idx}_{jdx}.png")
        cv2.imwrite(crop_filename, thresh)
        print(f"Saved crop: {crop_filename}")

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
