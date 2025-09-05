from rembg import new_session, remove
from PIL import Image

# input_path = input(r"C:\Users\ABC\Documents\receiptYOLOProject\IMG_0941.jpg")
# output_path = input(r"C:\Users\ABC\Documents\receiptYOLOProject\IMG_0941_.jpg")

input_path = r"C:\Users\ABC\Documents\receiptYOLOProject\IMG_0942.jpg"
output_path = r"C:\Users\ABC\Documents\receiptYOLOProject\IMG_0942_.jpg"

# Load original image and remove background
image = Image.open(input_path).convert("RGBA")
session = new_session("isnet-general-use")
foreground = remove(image, session=session)

# Create blue background image (same size)
blue_bg = Image.new("RGBA", foreground.size, (168, 231, 255, 255))  # (R, G, B, A)
white_bg = Image.new("RGBA", foreground.size, (255, 255, 255, 255))  # Pure white

# Composite foreground over blue background
composited = Image.alpha_composite(white_bg, foreground)

# Convert to RGB if saving as JPEG
if output_path.lower().endswith(".jpg") or output_path.lower().endswith(".jpeg"):
    composited = composited.convert("RGB")

# composited = composited.convert("RGB")
# composited.save("final_id_photo.jpg")

# Save final image
composited.save(output_path)
print(f"✅ Saved image with blue background to {output_path}")
