import cv2,os,glob
import numpy as np

# Read image
image_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\2025_0822_bad_cnndata\cnndata\images\tmp"

def rotate(deg=0.0,image_path=""):
    image = cv2.imread(image_path)
    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix (positive angle = counter-clockwise)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)  # angle=2 degrees, scale=1.0

    # Apply rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # Save or show
    return rotated
   

# Get all image files in the folder (jpg, png, etc.)
# Match both upper/lowercase JPG/PNG
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
image_files += glob.glob(os.path.join(image_dir, "*.jpeg"))
image_files += glob.glob(os.path.join(image_dir, "*.png"))
rotate_degrees = [-5.0,-3.0, -1.0, 1.0, 3.0,5.0]

for img_path in image_files:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    for idx,r in enumerate(rotate_degrees):
        rotated = rotate(r, img_path)
        if rotated is not None:
            save_path = os.path.join(image_dir, f"{filename}_rot{idx}_{r}.jpg")
            cv2.imwrite(save_path, rotated)
            print(f"✅ Saved {save_path}")

