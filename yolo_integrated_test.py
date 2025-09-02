from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO


image_path = r"C:\Users\ABC\Documents\receiptYOLOProject\test53.jpg"
image = cv2.imread(image_path)

# Path to your model
crop_model_path = r"C:\Users\ABC\Documents\receiptYOLOProject\v5_best_totalPairs.pt"
crop_model = YOLO(crop_model_path)
crop_results = crop_model(source=image, conf=0.01, save=True, show=True)

for idx, crop_result in enumerate(crop_results):
    for jdx, total_box in enumerate(crop_result.boxes):
        cls_id = int(total_box.cls[0])  # get class index as int
        if cls_id != 1:   # skip anything not class 1
            continue

        x_min, y_min, x_max, y_max = map(int, total_box.xyxy[0].tolist())
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


        # Load the model
        parse_model_path = r"C:\Users\ABC\Documents\receiptYOLOProject\crnn_model_8k_valloss_0.19_stable.keras"
        parse_model = load_model(parse_model_path, compile=False)
        img = thresh.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)            # add channel dimension (H x W x 1)
        img = np.expand_dims(img, axis=0)             # add batch dimension (1 x H x W x 1)

        preds = parse_model.predict(img)
        decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)

        decoded_indices = decoded[0].numpy()[0]
        decoded_text = [idx_to_char[i] for i in decoded_indices if i > 0]  # skip 0 and negatives
        print("Decoded:", decoded_text)
        decoded_text = ''.join(decoded_text)
        


