from PIL import Image, ImageDraw, ImageFont
import random
import os

# def generate_text_image(text, font_path="arial.ttf", size=32):
    # # Create a blank white image
    # img = Image.new("L", (200, size), color=255)
    # draw = ImageDraw.Draw(img)
    
    # # Pick a font
    # font = ImageFont.truetype(font_path, size-4)
    
    # # Draw text
    # draw.text((5, 0), text, font=font, fill=0)
    
    # # Random distortions (optional)
    # # img = img.rotate(random.randint(-2, 2), expand=True, fillcolor=255)
    
    # return img

def generate_text_image(text, font_path="arial.ttf", size=32, margin=5, max_width=200):
    font = ImageFont.truetype(font_path, size-4)
    
    # Create dummy image for measurement
    dummy_img = Image.new("L", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    # Get bounding box of text (left, top, right, bottom)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Add margin
    width = text_width + margin * 2
    height = size
    
    # Clamp to max width
    width = min(width, max_width)
    
    # Create final image
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((margin, 0), text, font=font, fill=0)
    
    return img


vocab = ['8', '7', '6', '3', '5', '4','1','2','9','0']
vocab2 = [':', '/', '-']

output_dir = r"C:\Users\ABC\Documents\clean_unique\synthetic"
os.makedirs(output_dir, exist_ok=True)

for i in range(500):
    # Generate year or time stamplike str
    tstr = ""
    tstr += "".join(random.choice(vocab) for _ in range(4))
    
    tstr += random.choice(vocab2)
    
    tstr += "".join(random.choice(vocab) for _ in range(2))

    tstr += random.choice(vocab2)

    tstr += "".join(random.choice(vocab) for _ in range(2))
    
    # Create image
    img = generate_text_image(tstr)

    fname = tstr.replace("/","sl")
    fname = fname.replace(":",";")
    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic.png"))

for i in range(500):
    # Generate year or time stamplike str
    tstr = ""
    tstr += tstr.join(random.choice(vocab) for _ in range(2))
    
    tstr += random.choice(vocab2)
    
    tstr += "".join(random.choice(vocab) for _ in range(2))

    tstr += random.choice(vocab2)

    tstr += "".join(random.choice(vocab) for _ in range(2))
    
    # Create image
    img = generate_text_image(tstr)

    fname = tstr.replace("/","sl")
    fname = fname.replace(":",";")
    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic.png"))


for i in range(500):
    # Generate year or time stamplike str
    tstr = "".join(random.choice(vocab) for _ in range(2))

    tstr += random.choice(vocab2)

    tstr += "".join(random.choice(vocab) for _ in range(2))
    
    # Create image
    img = generate_text_image(tstr)

    fname = tstr.replace("/","sl")
    fname = fname.replace(":",";")
    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic_date.png"))


for i in range(500):
    # Generate year or time stamplike str
    tstr = "$"
    tstr += "".join(random.choice(vocab) for _ in range(2))

    tstr += "."

    tstr += "".join(random.choice(vocab) for _ in range(2))
    
    # Create image
    img = generate_text_image(tstr)

    fname = tstr.replace("/","sl")
    fname = fname.replace(":",";")
    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic_dollor.png"))


for i in range(500):
    # Generate year or time stamplike str
    tstr = ""
    tstr += "".join(random.choice(vocab) for _ in range(4))

    tstr += "年"

    tstr += "".join(random.choice(vocab) for _ in range(2))

    tstr += "月"

    tstr += "".join(random.choice(vocab) for _ in range(2))

    tstr += "日"
    
    # Create image
    img = generate_text_image(tstr)

    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic_jp_date.png"))

