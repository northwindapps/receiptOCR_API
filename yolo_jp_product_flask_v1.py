from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import cv2
import os
import datetime
import json

# --- Flask app ---
app = Flask(__name__)

# --- Directories ---
BASE_DIR = "/app"
SAVE_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load models once at startup ---
parse_model_path = os.path.join(BASE_DIR, "crnn_model_jpy_valloss0.2_10k.keras")
crop_model_path = os.path.join(BASE_DIR, "date_telephone_jp_best.pt")
crop_model_path2 = os.path.join(BASE_DIR, "text_chunk_epoch40_best.pt")

print("Loading models...")
parse_model = load_model(parse_model_path, compile=False)
crop_model_tel_date = YOLO(crop_model_path)
crop_model_all = YOLO(crop_model_path2)

# --- Helper: decode with CRNN ---
def decode_with_crnn(img, parse_model):
    vocab = "0123456789.,-$;/@￥年月日円() "
    idx_to_char = {i+1: c for i, c in enumerate(vocab)}

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    preds = parse_model.predict(img)
    decoded, _ = tf.keras.backend.ctc_decode(
        preds,
        input_length=np.ones(preds.shape[0]) * preds.shape[1],
        greedy=True
    )
    decoded_indices = decoded[0].numpy()[0]
    decoded_text = [idx_to_char.get(int(i), "?") for i in decoded_indices if i > 0]
    return "".join(decoded_text)

# --- Flask route ---
@app.route("/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image temporarily
    img_path = os.path.join(SAVE_DIR, file.filename)
    file.save(img_path)

    # Read with OpenCV
    image = cv2.imread(img_path)
    if image is None:
        return jsonify({"error": "Could not read image"}), 400

    results = []

    # Run YOLO models
    crop_tel_date_results = crop_model_tel_date(source=image, conf=0.1, save=False, show=False)
    crop_all_results = crop_model_all(source=image, conf=0.1, save=False, show=False)

    # --- Loop 1: Processing Telephone and Date crops ---
    print("\n--- Processing Telephone/Date Crops ---")
    all_boxes_tel_date = [
        (idx, jdx, box, "tel_date") 
        for idx, crop_result in enumerate(crop_tel_date_results)
        for jdx, box in enumerate(crop_result.boxes)
    ]

    datestr = ""
    telstr = ""

    for idx, jdx, total_box,kind in all_boxes_tel_date:
        x_min, y_min, x_max, y_max = map(int, total_box.xyxy[0].tolist())
        crop = image[y_min:y_max, x_min:x_max]  

        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=4
        )

        h, w = thresh.shape[:2]
        if w == 0 or h == 0: continue

        if w/h < 2.0: 
            continue

        target_h = 31
        scale = target_h / h
        w = int(w * scale)
        thresh = cv2.resize(thresh, (w, target_h))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 55:
                cv2.drawContours(thresh, [cnt], -1, 0, -1)

        vocab = "0123456789.,-$;/@￥年月日円() "
        idx_to_char = {i+1:c for i,c in enumerate(vocab)}

        img = thresh.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        
        preds = parse_model.predict(img)
        decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)
        decoded_indices = decoded[0].numpy()[0]
        decoded_text = [idx_to_char.get(int(i), "?") for i in decoded_indices if i > 0]
        decoded_str = "".join(decoded_text)
        print("Decoded:", decoded_str)
        results.append({"kind": kind, "text": decoded_str})

        dateary = decoded_str.split("-")
        if len(dateary) == 3:
            telstr = decoded_str
        if "年" in decoded_str and "月" in decoded_str and "日" in decoded_str:
            datestr = decoded_str
        if telstr and datestr:
            break

        

    # --- Loop 2: Processing All Text crops ---
    print("\n--- Processing All Text Crops ---")
    all_boxes_text = [
        (idx, jdx, box, "all_boxes") 
        for idx, crop_result in enumerate(crop_all_results)
        for jdx, box in enumerate(crop_result.boxes)
    ]

    for idx, jdx, total_box,kind in all_boxes_text:
        x_min, y_min, x_max, y_max = map(int, total_box.xyxy[0].tolist())
        crop = image[y_min:y_max, x_min:x_max]  

        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=4
        )

        h, w = thresh.shape[:2]
        if w == 0 or h == 0: continue

        if w/h > 5.0:
            print(f"Ignoring wide crop (w/h > 5.0): {w/h:.2f}")
            continue
        if w/h < 2.0: 
            continue

        target_h = 31
        scale = target_h / h
        w = int(w * scale)
        thresh = cv2.resize(thresh, (w, target_h))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 55:
                cv2.drawContours(thresh, [cnt], -1, 0, -1)

        vocab = "0123456789.,-$;/@￥年月日円() "
        idx_to_char = {i+1:c for i,c in enumerate(vocab)}

        img = thresh.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        
        preds = parse_model.predict(img)
        decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)
        decoded_indices = decoded[0].numpy()[0]
        decoded_text = [idx_to_char.get(int(i), "?") for i in decoded_indices if i > 0]
        decoded_str = "".join(decoded_text)
        print("Decoded:", decoded_str)
        results.append({"kind": kind, "text": decoded_str})
        if "@" in decoded_str:
            break

    # return jsonify({"results": results})
    return app.response_class(
        response=json.dumps({"results": results}, ensure_ascii=False),
        status=200,
        mimetype="application/json"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
