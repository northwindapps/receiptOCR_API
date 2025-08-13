import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh_to_xyxy(box):
    """Convert [center_x, center_y, width, height] to [x_min, y_min, x_max, y_max]."""
    cx, cy, w, h = box
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return [x_min, y_min, x_max, y_max]

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        rest = indices[1:]

        filtered = []
        for i in rest:
            if iou(boxes[current], boxes[i]) < iou_threshold:
                filtered.append(i)
        indices = np.array(filtered)
    return keep

def process_output(output, image_size, conf_threshold=0.5):
    output = output[0]  # remove batch dim
    num_preds = output.shape[1]
    num_classes = output.shape[0] - 4

    boxes = []
    scores = []
    class_ids = []

    img_w, img_h = image_size

    for i in range(num_preds):
        cx, cy, w, h = output[0:4, i]
        conf = output[4, i]
        class_probs = output[5:, i]
        class_id = np.argmax(class_probs)
        class_score = class_probs[class_id] * conf

        if class_score < conf_threshold:
            continue

        # The model input size is fixed (640x640), but we want to scale to original image size:
        scale_x = img_w / 640
        scale_y = img_h / 640

        box = xywh_to_xyxy([cx * 640 * scale_x, cy * 640 * scale_y, w * 640 * scale_x, h * 640 * scale_y])

        boxes.append(box)
        scores.append(class_score)
        class_ids.append(class_id)

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    keep = non_max_suppression(boxes, scores)

    results = []
    for idx in keep:
        results.append({
            "box": boxes[idx].tolist(),
            "score": float(scores[idx]),
            "class_id": int(class_ids[idx])
        })

    return results

# Load model and image
model_path = r"C:\Users\ABC\Documents\receiptYOLOProject\layout-annotation-v2yolo11_tf_web_model\content\runs\detect\train\weights\best_saved_model\best_float32.tflite"
image_path = r"C:\Users\ABC\Documents\receiptYOLOProject\X00016469670.jpg"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open(image_path).convert("RGB")
img_w, img_h = img.size
img_resized = img.resize((640, 640))
input_data = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

print(img_w)
print(img_h)

results = process_output(output_data, image_size=(img_w, img_h), conf_threshold=0.25)

# Draw boxes
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except IOError:
    font = ImageFont.load_default()

for res in results:
    box = res['box']
    class_id = res['class_id']
    score = res['score']

    x_min, y_min, x_max, y_max = box
    # Draw rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    # Draw label
    text = f"Class {class_id}: {score:.2f}"
    draw.text((x_min, y_min - 15), text, fill="red", font=font)

img.show()  # Opens the image with default viewer

# Optionally save the image
img.save("output_with_boxes.jpg")
