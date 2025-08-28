import cv2,os,glob,datetime
import numpy as np

# Read image
# image_dir = r"C:\Users\ABC\Documents\clean_unique\clean"
image_dir = r"C:\Users\ABC\Documents\dirty_unique\dirty"
# output_dir = r"C:\Users\ABC\Documents\clean_unique\output"
output_dir = r"C:\Users\ABC\Documents\dirty_unique\output"
def rotate(deg=0.0,image=None):
    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix (positive angle = counter-clockwise)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)  # angle=2 degrees, scale=1.0

    # Apply rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # Save or show
    return rotated

def margin(m=0, image_path=""):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to process {img_path}")
            return None
        print(f"Processed {img_path} successfully.")
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")
    
    margin = m
    # Get image dimensions
    (h, w) = image.shape[:2]
    if len(image.shape) == 2:  # grayscale
        chunk_crop_with_bg = cv2.copyMakeBorder(
            image, margin, margin, margin, margin,
            cv2.BORDER_CONSTANT, value=255
        )
    else:  # color (3 channels)
        chunk_crop_with_bg = cv2.copyMakeBorder(
            image, margin, margin, margin, margin,
            cv2.BORDER_CONSTANT, value=[255,255,255]
        )
    return chunk_crop_with_bg
   

# Get all image files in the folder (jpg, png, etc.)
# Match both upper/lowercase JPG/PNG
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
image_files += glob.glob(os.path.join(image_dir, "*.jpeg"))
image_files += glob.glob(os.path.join(image_dir, "*.png"))
rotate_degrees = [-3.0,-1.5, 1.5,3.0]
margins = [0,3,6]
beta = [0]

for img_path in image_files:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    for m in margins:
        marginadded = margin(m, img_path)
        if marginadded is None:
            print("None in margin op")
            continue
        for idx,r in enumerate(rotate_degrees):
            rotated = rotate(r, marginadded)
            for bv in beta:
                contrast = cv2.convertScaleAbs(rotated, alpha=1.0, beta=bv)
                if contrast is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    save_path = os.path.join(output_dir, f"{filename}_rot{idx}_{r}_{m}_{bv}_{timestamp}.jpg")
                    cv2.imwrite(save_path, contrast)
                    print(f"✅ Saved {save_path}")

