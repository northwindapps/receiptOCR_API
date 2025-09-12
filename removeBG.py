import os
from rembg import new_session, remove
from PIL import Image

# input_path = input(r"C:\Users\ABC\Documents\receiptYOLOProject\IMG_0968.jpg")
# output_path = input(r"C:\Users\ABC\Documents\receiptYOLOProject\IMG_0968_.jpg")


# Paths
input_path = r"C:\Users\ABC\Documents\receiptYOLOProject\input"
output_path = r"C:\Users\ABC\Documents\receiptYOLOProject\output"

# Make sure output folder exists
os.makedirs(output_path, exist_ok=True)

# Load background removal session once (faster than reloading for every image)
session = new_session("isnet-general-use")

# Process each file in the input folder
for filename in os.listdir(input_path):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue  # skip non-image files

    input_file = os.path.join(input_path, filename)
    output_file = os.path.join(output_path, filename)

    print(f"⏳ Processing {filename}...")

    # Load original image
    image = Image.open(input_file).convert("RGBA")

    # Remove background
    foreground = remove(image, session=session)

    # Create backgrounds
    white_bg = Image.new("RGBA", foreground.size, (255, 255, 255, 255))  

    # Composite foreground over background
    composited = Image.alpha_composite(white_bg, foreground)

    # Convert to RGB if saving as JPEG
    if output_file.lower().endswith((".jpg", ".jpeg")):
        composited = composited.convert("RGB")

    # Save
    composited.save(output_file)
    print(f"✅ Saved to {output_file}")