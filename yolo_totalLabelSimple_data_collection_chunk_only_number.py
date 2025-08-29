import cv2
import pytesseract
from ultralytics import YOLO
import os,re
import easyocr
import datetime
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

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
    # if x_min < sharp_w/3:
    #     return False
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

                        if w/h > 6.0:
                            break

                        target_h = 31

                        # scale based on height
                        if h > target_h or h < target_h:
                            scale = target_h / h
                            w = int(w * scale)
                            h = target_h
                            thresh = cv2.resize(thresh, (w, target_h))

                        # Remove small blobs
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < cont_value:
                                cv2.drawContours(thresh, [cnt], -1, 0, -1)
                        vocab = "0123456789.,-{}()$￥@;~ " # Added space to vocab
                        char_to_idx = {c:i+1 for i,c in enumerate(vocab)}  # 0 reserved for blank
                        idx_to_char = {i+1:c for i,c in enumerate(vocab)}
                        # Path to your model
                        model_path = r"C:\Users\ABC\Documents\receiptYOLOProject\crnn_model_15k_good.h5"

                        # Load the model
                        base_model = load_model(model_path, compile=False)
                        img = thresh.astype(np.float32) / 255.0
                        img = np.expand_dims(img, axis=-1)            # add channel dimension (H x W x 1)
                        img = np.expand_dims(img, axis=0)             # add batch dimension (1 x H x W x 1)


                        preds = base_model.predict(img)
                        decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)

                        decoded_indices = decoded[0].numpy()[0]
                        decoded_text = [idx_to_char[i] for i in decoded_indices if i > 0]  # skip 0 and negatives
                        print("Decoded:", decoded_text)
                        decoded_text = ''.join(decoded_text)

                         # Save the cropped image
                        crop_filename = os.path.join(save_original_dir, f"0_{decoded_text}_{idx}_{jdx}_original.png")
                        cv2.imwrite(crop_filename, layout_crop)
                        print(f"Saved crop: {crop_filename}")


                        # Run Tesseract
                        data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
                        avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
                        avg_conf /= 100

                        text = pytesseract.image_to_string(g, lang='eng', config="--psm 7 -c tessedit_char_whitelist=0123456789:.$").strip()
                        # text = pytesseract.image_to_string(thresh, lang='eng').strip()

                        if text.strip() == '':
                            return False
                        
                        if len(text.strip()) < 5:
                            print("not 4 digits")
                            return False

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
                        if best_conf < 0.5:
                            return False
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
                            crop_filename = os.path.join(save_dir, f"90_{safe_filename(best_text)}_{safe_filename(decoded_text)}_{timestamp}.jpg")
                            # cv2.imwrite(crop_filename, thresh)
                            cv2.imwrite(crop_filename,thresh,[int(cv2.IMWRITE_JPEG_QUALITY), 95] )
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
                                crop_filename = os.path.join(save_dir, f"70_{safe_filename(best_text)}_{safe_filename(decoded_text)}_{timestamp}.jpg")
                                cv2.imwrite(crop_filename,thresh,[int(cv2.IMWRITE_JPEG_QUALITY), 95] )
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
# image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\IMG_0955.jpg'
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test2.jpg'
image = cv2.imread(image_path)
sharpened = image

# Get image width
img_height, img_width = image.shape[:2]

# Create folder to save crops
save_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\dataset\crops"
os.makedirs(save_dir, exist_ok=True)

save_original_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\dataset\crops\original"
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
     
        