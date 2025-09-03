from difflib import SequenceMatcher
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2,os,datetime
from ultralytics import YOLO
import pytesseract

def is_total(word):
    word = word.strip().upper()
    return SequenceMatcher(None, word, "TOTAL").ratio() > 0.7  # threshold 0.7-0.8

# Need to make this script class 
def detect_total_labels(image):
    total_label_cords = []
    print(f"\n=== Detecting Label in image with Pytesseract===")
    layout_crop = image
    gray = cv2.cvtColor(layout_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresh, lang='eng').strip()
    data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data['conf'] if int(c) >= 0]
    avg_conf = sum(confidences) / max(1, len(confidences)) / 100

    if text and avg_conf > 0.5:
        # print(f"OCR Confidence: {avg_conf:.2f}")
        # print(f"Chunk Text by Pytesseract: {text}")
        # print("=" * 40)
        for i, word in enumerate(data['text']):
            if is_total(word):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                # print(f"Found TOTAL at: x={x}, y={y}, w={w}, h={h}, conf={data['conf'][i]}")
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.imwrite("receipt_with_total.jpg", image)
                total_label_cords.append([x, y, x+w, y+h])
    return total_label_cords

def parse_boxs(image,total_label_cords):
    for kdx, label_cord in enumerate(total_label_cords):
        # parse back
        x_min_l, y_min_l, x_max_l, y_max_l = map(int, label_cord)

        # if abs(y_min_l - y_min) < 150 and abs(y_max_l - y_max) < 150:
        if abs(y_min_l - y_min) < 50:

            crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]  

            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                    
            thresh = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)

            h, w = thresh.shape[:2]

            if w/h > 6.0:
                print("you can ignore it w/h>6.0,it's too wide")
                
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
                if cv2.contourArea(cnt) < 55:
                    cv2.drawContours(thresh, [cnt], -1, 0, -1)
            vocab = "0123456789.,-$ " # Added space to vocab
            char_to_idx = {c:i+1 for i,c in enumerate(vocab)}  # 0 reserved for blank
            idx_to_char = {i+1:c for i,c in enumerate(vocab)}

            img = thresh.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)            # add channel dimension (H x W x 1)
            img = np.expand_dims(img, axis=0)             # add batch dimension (1 x H x W x 1)
            preds = parse_model.predict(img)
            decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)

            decoded_indices = decoded[0].numpy()[0]
            # decoded_text = [idx_to_char[i] for i in decoded_indices if i > 0]  # skip 0 and negatives
            decoded_text = [idx_to_char.get(int(i), "?") for i in decoded_indices if i > 0]
            print("Decoded:", decoded_text)
            decoded_text = ''.join(decoded_text)
            try:
                float_cast = float(decoded_text)   # or float(decoded_text)
            except ValueError:
                float_cast = None  # or handle invalid cases
            if float_cast is not None:
                total_value_texts.append(float_cast)
            print(total_value_texts)
            # Save the cropped image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_filename = os.path.join(save_original_dir, f"0_{decoded_text}_{idx}_{jdx}_{timestamp}_original.png")
            cv2.imwrite(crop_filename, thresh)
            print(f"Saved crop: {crop_filename}")


save_original_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\dataset\crops\all"
os.makedirs(save_original_dir, exist_ok=True)

# Need preprocess for dark image...
image_path = r"C:\Users\ABC\Documents\receiptYOLOProject\test55.jpg"
image = cv2.imread(image_path)

# Load the models
parse_model_path = r"C:\Users\ABC\Documents\receiptYOLOProject\crnn_model_8k_valloss_0.19.h5"
parse_model = load_model(parse_model_path, compile=False)
crop_model_path = r"C:\Users\ABC\Documents\receiptYOLOProject\v5_best_totalPairs.pt"
crop_model = YOLO(crop_model_path)
# Definately need drastic improvment!
crop_results = crop_model(source=image, conf=0.1, save=True, show=True)

# Pytesseract Layer
total_label_cords = []
# --- Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
total_label_cords = detect_total_labels(image)
print('TOTAL_COORDINATES',total_label_cords)

# Checking the first layer Results
total_value_texts = []
total_label_cords_bk = []
for idx, crop_result in enumerate(crop_results):
    for jdx, total_box in enumerate(crop_result.boxes):
        x_min, y_min, x_max, y_max = map(int, total_box.xyxy[0].tolist())
        cls_id = int(total_box.cls[0])  # get class index as int
        if cls_id != 1:   # skip anything not class 1
            total_label_cords_bk.append([x_min, y_min, x_max, y_max])
            continue
        if total_label_cords:
            parse_boxs(image,total_label_cords)

            

#Move to backup layer            
# if not total_label_cords and not total_value_texts: which should I choose?
if total_label_cords and not total_value_texts:
    for idx, crop_result in enumerate(crop_results):
        for jdx, total_box in enumerate(crop_result.boxes):
            x_min, y_min, x_max, y_max = map(int, total_box.xyxy[0].tolist())
            cls_id = int(total_box.cls[0])  # get class index as int
            if cls_id != 1:   # skip anything not class 1
                continue
            parse_boxs(image,total_label_cords_bk)