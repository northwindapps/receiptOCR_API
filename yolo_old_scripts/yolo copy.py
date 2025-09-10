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


import cv2
import numpy as np
from scipy.ndimage import white_tophat
from skimage import morphology

def enhance_receipt(image_path):
    """
    Complete pipeline for receipt enhancement
    """
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Shadow Removal (Rolling Ball Algorithm)
    shadow_removed = remove_shadows(gray)
    
    # 2. Contrast Enhancement
    enhanced = enhance_contrast(shadow_removed)
    
    # 3. Handle crumpled/folded areas
    dewarped = handle_distortions(enhanced)
    
    # 4. Final binarization for OCR
    final = prepare_for_ocr(dewarped)
    
    return final

def remove_shadows(image):
    """
    Remove uneven lighting and shadows
    """
    # Rolling ball algorithm
    dilated = cv2.dilate(image, np.ones((7,7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    
    # Remove background
    diff = 255 - cv2.absdiff(image, bg)
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized

def enhance_contrast(image):
    """
    Adaptive contrast enhancement
    """
    # CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Sharpen text
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def handle_distortions(image):
    """
    Handle crumpled or folded receipts
    """
    # Morphological operations to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    denoised = cv2.fastNlMeansDenoising(closed, h=10)
    
    return denoised

def prepare_for_ocr(image):
    """
    Final preparation for OCR
    """
    # Adaptive thresholding for text
    binary = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Remove small specs
    cleaned = remove_small_objects(binary)
    
    return cleaned

def remove_small_objects(binary_image, min_size=50):
    """
    Remove small noise specs
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    
    # Filter small components
    sizes = stats[1:, -1]
    num_labels = num_labels - 1
    
    cleaned = np.zeros_like(binary_image)
    for i in range(num_labels):
        if sizes[i] >= min_size:
            cleaned[labels == i + 1] = 255
            
    return cleaned

# Advanced shadow removal for severe cases
def advanced_shadow_removal(image):
    """
    For receipts with heavy shadows
    """
    # Estimate illumination using morphological operations
    struct_elem = morphology.disk(20)
    background = white_tophat(image, struct_elem)
    