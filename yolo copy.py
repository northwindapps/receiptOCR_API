import cv2
import pytesseract
import ultralytics
from ultralytics import YOLO

model = YOLO('layout_best.pt')
second_model = YOLO('text_chunk_best.pt')
results = model(source=r'C:\Users\ABC\Documents\receiptYOLOProject\X00016469670.jpg',show=True, conf=0.25, save=True)
#results = model(source=r'C:\Users\ABC\Documents\receiptYOLOProject\shipping_label_02.png',show=True, conf=0.25, save=True)

# results is a list (one per image)
# for result in results:
#     boxes = result.boxes  # Boxes object

#     # xyxy coordinates (x1, y1, x2, y2)
#     for box in boxes:
#         coords = box.xyxy[0].tolist()   # convert tensor to Python list
#         conf = float(box.conf[0])       # confidence score
#         cls = int(box.cls[0])           # class index
#         print(f"Class {cls} | Conf {conf:.2f} | BBox {coords}")

#         # If you want xywh instead:
#         # coords = box.xywh[0].tolist()

# Load the image with OpenCV
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\X00016469670.jpg';
image = cv2.imread(image_path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR each box
for result in results:
    for box in result.boxes:
        # Get coordinates in xyxy format
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

        # Crop the region
        cropped = image[y_min:y_max, x_min:x_max]

        # Optional: convert to grayscale and threshold for better OCR
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Run Tesseract
        text = pytesseract.image_to_string(thresh, lang='eng')

        # Print results
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {cls_id}, Conf: {conf:.2f}")
        print("OCR Text:", text.strip())
        print("="*50)