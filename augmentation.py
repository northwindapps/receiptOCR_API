import cv2, os, glob, numpy as np, datetime

# Read image
# image_dir = r"C:\Users\ABC\Documents\clean_unique\clean"
# image_dir = r"C:\Users\ABC\Documents\dirty_unique\dirty"
# image_dir = r"C:\Users\ABC\Documents\cnndata\images"
# # output_dir = r"C:\Users\ABC\Documents\clean_unique\output"
# # output_dir = r"C:\Users\ABC\Documents\dirty_unique\output"
# output_dir = r"C:\Users\ABC\Documents\cnndata\output"


# Directories
image_dir = r"C:\Users\ABC\Documents\clean_unique\clean"
output_dir = r"C:\Users\ABC\Documents\clean_unique\\output"
os.makedirs(output_dir, exist_ok=True)

# Augmentation parameters
rotate_range = (-3.0, 3.0)   # random rotation in degrees
margin_range = (0, 6)        # random padding
contrast_range = (0.9, 1.1)  # alpha
brightness_range = (-10, 10) # beta
noise_std = 5                # Gaussian noise standard deviation
aug_per_image = 4            # number of augmentations per image

# Helper functions
def rotate(image, deg):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

def add_margin(image, m):
    return cv2.copyMakeBorder(image, m, m, m, m, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def adjust_contrast_brightness(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_noise(image, std):
    noise = np.random.normal(0, std, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

# Get all images
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
image_files += glob.glob(os.path.join(image_dir, "*.jpeg"))
image_files += glob.glob(os.path.join(image_dir, "*.png"))

# Augmentation loop
for img_path in image_files:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        continue

    for i in range(aug_per_image):
        m = np.random.randint(*margin_range)
        rotated = rotate(add_margin(img, m), np.random.uniform(*rotate_range))
        alpha = np.random.uniform(*contrast_range)
        beta = np.random.randint(*brightness_range)
        augmented = adjust_contrast_brightness(rotated, alpha, beta)
        # augmented = add_noise(augmented, noise_std)  # optional, comment out if not needed

        save_path = os.path.join(output_dir, f"{filename}_aug{i}_m{m}_a{alpha:.2f}_b{beta}.jpg")
        cv2.imwrite(save_path, augmented)
        print(f"✅ Saved {save_path}")
