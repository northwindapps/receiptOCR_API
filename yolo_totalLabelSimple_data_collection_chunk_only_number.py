import cv2
import pytesseract
from ultralytics import YOLO
import os,re
import easyocr
import datetime
import numpy as np

def rotate(img,v):
    # Get image dimensions
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix (positive angle = counter-clockwise)
    M = cv2.getRotationMatrix2D(center, v, 1.0)  # angle=2 degrees, scale=1.0

    # Apply rotation
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    return rotated
def safe_filename(text):
    # Replace forbidden characters with underscore
    return re.sub(r'[\\/*?:"<>|]', "_", text)

def reading_part(rotate_degrees,idx,jdx,cont_area_values, sharpend,totalLabel_box,alpha_values,beta_values,scales):
    # Get current datetime string (YYYYMMDD_HHMMSS format)
    target_conf = 0.9
    best_text = ""
    best_conf = 0
    prev = ""
    count = 0
    count_50 = 0
    count_15 = 0
    x_min, y_min, x_max, y_max = map(int, totalLabel_box.xyxy[0].tolist())
    # Loop through margin adjustments
    sharp_w,sharp_h, _ = sharpend.shape
    if x_min < sharp_w/3:
        return False
    for r in [0.0]:
        rotated = rotate(sharpend,r)
        # Loop through all alpha/beta combinations
        for cont_value in cont_area_values:
            layout_crop = rotated[int(y_min):int(y_max), int(x_min):int(x_max)]    
            for av in alpha_values:
                print(f"alpha{av}")
                for scale in scales:
                    print(f"scale {scale}")
                    for bv in beta_values:

                        # rotated = rotate(layout_crop,degree)
                        # contrast = cv2.convertScaleAbs(rotated, alpha=av, beta=bv)
                        contrast = layout_crop
                        processed_img = cv2.resize(contrast, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        g = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                        
                        thresh = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)

                        h, w = thresh.shape[:2]

                        target_w, target_h = 292, 62

                        # create white background
                        if len(thresh.shape) == 2:  # grayscale
                            canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
                        else:  # color
                            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

                        # offsets for centering (or left align if you prefer)
                        x_offset = (target_w - w) // 2  # use 0 for left align
                        y_offset = (target_h - h) // 2  # use 0 for top align

                        if x_offset >= 0 and y_offset >= 0 and (y_offset + h) <= target_h and (x_offset + w) <= target_w:
                            # paste the crop
                            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = thresh
                        else:
                            break

                        #keep the original before updating thresh
                        margin = 0
                        if h < 45:
                            margin = 3

                        if len(thresh.shape) == 2:  # grayscale
                            chunk_crop_with_bg = cv2.copyMakeBorder(
                                thresh, margin, margin, margin, margin,
                                cv2.BORDER_CONSTANT, value=255
                            )
                        else:  # color (3 channels)
                            chunk_crop_with_bg = cv2.copyMakeBorder(
                                thresh, margin, margin, margin, margin,
                                cv2.BORDER_CONSTANT, value=[255,255,255]
                            )

                        thresh = canvas

                        # Remove small blobs
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < cont_value:
                                cv2.drawContours(thresh, [cnt], -1, 0, -1)
                        # Save the cropped image
                        crop_filename = os.path.join(save_dir, f"totalLabel_crop_{idx}_{jdx}.png")
                        cv2.imwrite(crop_filename, chunk_crop_with_bg)
                        print(f"Saved crop: {crop_filename}")
                        # Run Tesseract
                        data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
                        avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
                        avg_conf /= 100

                        text = pytesseract.image_to_string(g, lang='eng', config="--psm 7 -c tessedit_char_whitelist=0123456789:.$").strip()
                        # text = pytesseract.image_to_string(thresh, lang='eng').strip()

                        if avg_conf > best_conf:
                            best_conf = avg_conf
                            best_text = text
                        # if avg_conf < 0.01:
                        #     # want to move next alpha
                        #     stop_early = True
                    
                        if best_conf >= 0.70 and prev == best_text:
                            count += 1
                        else:
                            prev = best_text
                            count = 0    
                        if (best_conf < 0.70 and best_conf >= 0.50) and prev == best_text:
                            count_50 += 1
                        else:
                            prev = best_text
                            count_50 = 0
                        # After trying all alpha/beta/margin combinations
                        if best_conf >= target_conf:
                            prev = ''
                            count = 0
                            count_50 = 0
                            print("easyOCR")
                            # Convert BGR to RGB for EasyOCR
                            chunk_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

                            # OCR with EasyOCR
                            results = reader.readtext(chunk_rgb, detail=1)
                            clean_text = ''
                            conf = 0.0
                            for bbox, text, confidence in results:
                                # print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")
                                conf = confidence
                                clean_text = text.replace("-", ".")
                                print(clean_text)
                                if any(char.isdigit() for char in clean_text) and not clean_text.endswith(".") and not clean_text.startswith("."):
                                    if confidence >= 0.9:
                                        print(f"Chunk Text: {clean_text}")
                                        print(f"Confidence: {confidence:.2f}")
                                        print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                        print("Contains numbers")
                                    if confidence >= 0.7 and "$" in clean_text:
                                        print(f"Chunk Text: {clean_text}")
                                        print(f"Confidence: {confidence:.2f}")
                                        print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                        print("Contains $")
                            # Save the cropped image
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                            crop_filename = os.path.join(save_dir, f"90_crop_pyttext_{safe_filename(best_text)}_easyOCR_text_{safe_filename(clean_text)}_ts_{timestamp}.jpg")
                            # cv2.imwrite(crop_filename, thresh)
                            cv2.imwrite(crop_filename,chunk_crop_with_bg,[int(cv2.IMWRITE_JPEG_QUALITY), 95] )
                            print(f"Saved crop: {crop_filename}")
                            print(f"alpha:beta:contour,scale: [{av}, {bv}, {cont_value},{scale}]")
                            count_15 += 1
                            if count_15 > 5:
                                return False
                            continue
                        else:
                            print(f"No high-confidence result found, best was: {best_text} ({best_conf:.2f})")
                            # stop_early = True
                            if any(char.isdigit() for char in best_text) and (count > 2 or count_50 > 3):
                                prev = ''
                                count = 0
                                count_50 = 0
                                
                                print("easyOCR")
                                # Convert BGR to RGB for EasyOCR
                                chunk_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

                                # OCR with EasyOCR
                                results = reader.readtext(chunk_rgb, detail=1)
                                clean_text = ''
                                conf = 0.0
                                for bbox, text, confidence in results:
                                    # print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")
                                    conf = confidence
                                    clean_text = text.replace("-", ".")
                                    print(clean_text)
                                    if any(char.isdigit() for char in clean_text) and not clean_text.endswith(".") and not clean_text.startswith("."):
                                        if confidence >= 0.9:
                                            print(f"Chunk Text: {clean_text}")
                                            print(f"Confidence: {confidence:.2f}")
                                            print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                            print("Contains numbers")
                                        if confidence >= 0.7 and "$" in clean_text:
                                            print(f"Chunk Text: {clean_text}")
                                            print(f"Confidence: {confidence:.2f}")
                                            print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                            print("Contains $")
                                # Save the cropped image
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                crop_filename = os.path.join(save_dir, f"70_crop_pyttext_{safe_filename(best_text)}_easyOCR_text_{safe_filename(clean_text)}_ts_{timestamp}.jpg")
                                # cv2.imwrite(crop_filename, thresh)
                                cv2.imwrite(crop_filename,chunk_crop_with_bg,[int(cv2.IMWRITE_JPEG_QUALITY), 95] )
                                print(f"Saved crop: {crop_filename}")
                                print(f"alpha:beta:contour,scale: [{av}, {bv}, {cont_value},{scale}]")
                                count_15 += 1
                                if count_15 > 5:
                                    return False
                                continue
                            
    return True

# Load models
totalLabel_model = YOLO('text_chunk_epoch40_best.pt')         

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test50.jpg'
image = cv2.imread(image_path)
sharpened = image

# Get image width
img_height, img_width = image.shape[:2]

# Create folder to save crops
save_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\dataset\crops"
os.makedirs(save_dir, exist_ok=True)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have a GPU and want to use it

 # Define contrast & brightness variations to try
# alpha_values = [-3.0,-1.0,1.5,3.0,5.0]  # contrast
# beta_values = [-80,-50,0,50,80,160]  # brightness
alpha_values = [-2.0,-1.0,1.0,2.0]  # contrast
beta_values = [100,50,25,0,-25,-50,-100]
cont_area_values = [15,55]
rotate_degrees = [-1.5,-1.0,0,1.0,1.5]
scales = [1.0,1.2,1.4,1.6,1.8]
# Run layout detection
totalLabel_results = totalLabel_model(source=sharpened, conf=0.10, save=True, show=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have a GPU and want to use it

for idx, totalLabel_result in enumerate(totalLabel_results):
    for jdx, totalLabel_box in enumerate(totalLabel_result.boxes):
        cls_id = int(totalLabel_box.cls[0])  # get class index as int
        # if cls_id != 1:   # skip anything not class 1
        #     continue
        if not reading_part(rotate_degrees=rotate_degrees,idx=idx,jdx=jdx,sharpend=sharpened,totalLabel_box=totalLabel_box,alpha_values=alpha_values,beta_values=beta_values,scales=scales,cont_area_values=cont_area_values):
            continue
     
        