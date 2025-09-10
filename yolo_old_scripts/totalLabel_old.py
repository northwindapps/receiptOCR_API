import cv2
import pytesseract
from ultralytics import YOLO
import os
import easyocr

# Load models
totalLabel_model = YOLO('totalLabel_best.pt')         

chunk_model = YOLO('text_chunk_epoch40_best.pt')

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test9.jpg'
image = cv2.imread(image_path)

alpha = 2.0  # contrast factor
beta = 50    # brightness offset (tweak if needed)

bright_contrast_image = cv2.convertScaleAbs(image * alpha + beta)

# Get image width
img_height, img_width = image.shape[:2]

# Create folder to save crops
save_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\crops"
os.makedirs(save_dir, exist_ok=True)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Run layout detection
totalLabel_results = totalLabel_model(source=bright_contrast_image, conf=0.50, save=True, show=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have a GPU and want to use it

for idx,totalLabel_result in enumerate(totalLabel_results):
    for jdx,totalLabel_box in enumerate(totalLabel_result.boxes):
        # Layout region coords
        x_min, y_min, x_max, y_max = map(int, totalLabel_box.xyxy[0].tolist())
        # Crop from totalLabel box x_min to the right edge
        layout_crop = bright_contrast_image[int(y_min)+0:int(y_max)-0, int(x_max):img_width]

        # img = cv2.imread('/mnt/data/totalLabel_crop_0_0.png', cv2.IMREAD_GRAYSCALE)
        # scale_factor = 3
        # resized = cv2.resize(layout_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        

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

                chunk_crop = layout_crop[cy_min:cy_max, cx_min:cx_max]  # crop first
                scale_factor = 3
                resized_chunk = cv2.resize(chunk_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


                margin = 20

                # Optional: make white background if original is grayscale
                if len(resized_chunk.shape) == 2:  # grayscale
                    chunk_crop_with_bg = cv2.copyMakeBorder(
                        resized_chunk, margin, margin, margin, margin,
                        cv2.BORDER_CONSTANT, value=255
                    )
                else:  # color image
                    chunk_crop_with_bg = cv2.copyMakeBorder(
                        resized_chunk, margin, margin, margin, margin,
                        cv2.BORDER_CONSTANT, value=[255, 255, 255]
                    )

                # Save the cropped image
                crop_filename = os.path.join(save_dir, f"totalLabel_crop_{idx}_{jdx}.png")
                cv2.imwrite(crop_filename, chunk_crop_with_bg)
                print(f"Saved crop: {crop_filename}")

                # Preprocess for OCR
                gray = cv2.cvtColor(chunk_crop_with_bg, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < 10:  # threshold tiny blobs
                        cv2.drawContours(thresh, [cnt], -1, 0, -1)  # fill with black


                # OCR
                text = pytesseract.image_to_string(thresh, lang='eng', config='--oem 1')

                # text = pytesseract.image_to_string(thresh, lang='eng').strip()
                data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
                avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
                avg_conf /= 100
                if text != '' and avg_conf > 0.5:
                    print(f"Layout Class: {int(totalLabel_box.cls[0])}, Conf: {float(totalLabel_box.conf[0]):.2f}")
                    print(f"Average OCR Confidence: {avg_conf:.2f}")
                    # print(f"Chunk Class: {int(chunk_box.cls[0])}, Conf: {float(chunk_box.conf[0]):.2f}")
                    print(f"Chunk Text: {text}")
                    print(f"Chunk BBox (original image coords): [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")
                    print("="*40)

                # Convert BGR to RGB for EasyOCR
                chunk_rgb = cv2.cvtColor(chunk_crop_with_bg, cv2.COLOR_BGR2RGB)

                # OCR with EasyOCR
                # results = reader.readtext(chunk_rgb, detail=1)

                # for bbox, text, confidence in results:
                #     print(f"Chunk Text: {text}")
                #     print(f"Confidence: {confidence:.2f}")
                #     # print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                #     print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")

                #     print("=" * 40)
