import cv2
import pytesseract
from ultralytics import YOLO

# Load models
layout_model = YOLO('layout_best.pt')         
chunk_model = YOLO('text_chunk_epoch40_best.pt')

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test7.jpg'
image = cv2.imread(image_path)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Run layout detection
layout_results = layout_model(image_path, conf=0.50, save=True, show=True)

for layout_result in layout_results:
    for layout_box in layout_result.boxes:
        # Layout region coords
        x_min, y_min, x_max, y_max = map(int, layout_box.xyxy[0].tolist())
        layout_crop = image[y_min:y_max, x_min:x_max]

        # Run second model on cropped layout region
        chunk_results = chunk_model(layout_crop, conf=0.3)

        for chunk_result in chunk_results:
            for chunk_box in chunk_result.boxes:
                # Chunk coords (relative to cropped layout region)
                cx_min, cy_min, cx_max, cy_max = map(int, chunk_box.xyxy[0].tolist())

                # Calculate chunk coords relative to original image
                abs_x_min = x_min + cx_min
                abs_y_min = y_min + cy_min
                abs_x_max = x_min + cx_max
                abs_y_max = y_min + cy_max

                chunk_crop = layout_crop[cy_min:cy_max, cx_min:cx_max]

                # Preprocess for OCR
                gray = cv2.cvtColor(chunk_crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                # OCR
                text = pytesseract.image_to_string(thresh, lang='eng').strip()
                data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
                avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
                avg_conf /= 100
                if text != '' and avg_conf > 0.6:
                    print(f"Layout Class: {int(layout_box.cls[0])}, Conf: {float(layout_box.conf[0]):.2f}")
                    print(f"Average OCR Confidence: {avg_conf:.2f}")
                    # print(f"Chunk Class: {int(chunk_box.cls[0])}, Conf: {float(chunk_box.conf[0]):.2f}")
                    print(f"Chunk Text: {text}")
                    print(f"Chunk BBox (original image coords): [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")
                    print("="*40)
