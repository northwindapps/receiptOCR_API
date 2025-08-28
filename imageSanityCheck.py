import json
import cv2
import random
import os

# Paths
folder = r"C:\Users\ABC\Documents\cnndata\images"
text_file = r"C:\Users\ABC\Documents\cnndata\labels.json"

# Load JSON
with open(text_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract image paths and labels
image_paths = [os.path.join(folder, item["filename"]) for item in data]
texts = [item["annotation"] for item in data]

# Show 10 random samples
for p, lbl in random.sample(list(zip(image_paths, texts)), 10):
    print("GT:", lbl)
    im = cv2.imread(p)
    if im is None:
        print("Can't open", p)
        continue
    cv2.imshow("sample", cv2.resize(im, (im.shape[1], im.shape[0])))
    cv2.waitKey(2500)  # or save to disk if headless

cv2.destroyAllWindows()
