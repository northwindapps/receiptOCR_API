import cv2
from ultralytics import YOLO
import easyocr

# Load models
layout_model = YOLO('layout_best.pt')
chunk_model = YOLO('text_chunk_epoch40_best.pt')

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test7.jpg'
image = cv2.imread(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have a GPU and want to use it

# Run layout detection
layout_results = layout_model(image_path, conf=0.30, save=True, show=True)

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

                # Convert BGR to RGB for EasyOCR
                chunk_rgb = cv2.cvtColor(chunk_crop, cv2.COLOR_BGR2RGB)

                # OCR with EasyOCR
                results = reader.readtext(chunk_rgb, detail=1)

                for bbox, text, confidence in results:
                    print(f"Chunk Text: {text}")
                    print(f"Confidence: {confidence:.2f}")
                    # print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                    print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")

                    print("=" * 40)
