import cv2
from paddleocr import PaddleOCR

# Initialize
ocr = PaddleOCR(lang='en')
    
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test7.jpg'
img = cv2.imread(image_path)
# Process image
result = ocr.predict(
    img,
    use_doc_orientation_classify=True,   # optionally detect document rotation
    text_det_thresh=0.5,                  # detection confidence threshold
    text_rec_score_thresh=0.5             # recognition confidence threshold
)

# result is a list of dicts, each with text, confidence, and bounding box
# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")

