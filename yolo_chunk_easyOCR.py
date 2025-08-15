import cv2
import easyocr
from ultralytics import YOLO

# Load models
chunk_model = YOLO('text_chunk_epoch40_best.pt')

# EasyOCR reader (English only)
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test16_1_cropped.jpg'
image = cv2.imread(image_path)

 # Crop chunk from original image
# chunk_crop = image[y_min:y_max, x_min:x_max]

# Parameters to tweak sharpness
strength = 1.5  # How much to boost edges (1.0 = no change)
blur_size = (0, 0)  # Let OpenCV figure size from sigma
sigma = 3  # How much to blur before subtracting

# Unsharp masking
blurred = cv2.GaussianBlur(image, blur_size, sigma)
sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)


# EasyOCR expects RGB images
chunk_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

results = reader.readtext(chunk_rgb, detail=1)

for bbox, text, confidence in results:
    print(f"Chunk Text: {text}")
    print(f"Confidence: {confidence:.2f}")
    # print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
    print("=" * 40)


# Run chunk detection on full image
chunk_results = chunk_model(image_path, conf=0.2, save=True, show=True)

for chunk_result in chunk_results:
    for chunk_box in chunk_result.boxes:
        # Chunk coords in original image
        x_min, y_min, x_max, y_max = map(int, chunk_box.xyxy[0].tolist())

        # Crop chunk from original image
        chunk_crop = image[y_min:y_max, x_min:x_max]

        # EasyOCR expects RGB images
        chunk_rgb = cv2.cvtColor(chunk_crop, cv2.COLOR_BGR2RGB)

        results = reader.readtext(chunk_rgb, detail=1)

        for bbox, text, confidence in results:
            print(f"Chunk Text: {text}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            print("=" * 40)
