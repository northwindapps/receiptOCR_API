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

from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont

def generate_jp_text_image(text, font_path="C:/Windows/Fonts/meiryo.ttc", size=20, margin=5, max_width=200):
    font = ImageFont.truetype(font_path, size)

    # Measure bbox
    dummy_img = Image.new("L", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    width = min(text_width + margin * 2, max_width)
    height = text_height + margin * 2

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    # Correct centering: shift by bbox[1] (the baseline offset)
    x = (width - text_width) // 2 - bbox[0]
    y = (height - text_height) // 2 - bbox[1]

    draw.text((x, y), text, font=font, fill=0)

    return img



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

output_dir = r"C:\Users\ABC\Documents\clean_unique_kanji\synthetic"
os.makedirs(output_dir, exist_ok=True)

for i in range(200):
    # Generate year or time stamplike str
    year = random.randint(2000, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # safe range

    # Build Japanese date string
    tstr = f"{year}年{month:02d}月{day:02d}日"

    # Create image
    img = generate_jp_text_image(tstr)

    fname = tstr.replace("年","yy").replace("月","mm").replace("日","dd")

    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic_jp_date.png"))


for i in range(200):
    # Generate year or time stamplike str
    year = random.randint(2000, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # safe range

    # Build Japanese date string
    tstr = f"{year}年{month:02d}月{day:02d}日"

    # Create image
    img = generate_jp_text_image(tstr)

    fname = tstr.replace("年","yy").replace("月","mm").replace("日","dd")

    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic_jp_date.png"))


for i in range(300):
    # Generate year or timestamp-like str
    price = random.randint(1, 10000)

    # Format with comma separator
    tstr = f"￥{price:,}"
    
    # Create image
    img = generate_jp_text_image(tstr)

    # Safe filename (replace ￥ with jpy)
    fname = tstr.replace("￥", "jpy")

    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic_jp_jpy.png"))


for i in range(300):
    # Generate year or time stamplike str
    tstr = ""

    tstr += str(i)

    tstr += " 円"
    
    # Create image
    img = generate_jp_text_image(tstr)

    fname = tstr.replace("円","yen")
    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic.png"))


for i in range(400):
    # Generate year or time stamplike str
    tstr = "("

    tstr += str(i)

    tstr += ")円"
    
    # Create image
    img = generate_jp_text_image(tstr)

    fname = tstr.replace("円","yen")
    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic.png"))

for i in range(300):
    # Generate year or time stamplike str

    tstr = "@"

    price = round(random.uniform(1, 1000), 2)

    tstr += str(price)

    tstr += ""
    
    # Create image
    # img = generate_jp_text_image(font_path="C:/Windows/Fonts/msgothic.ttc" ,text=tstr)
    img = generate_jp_text_image(tstr)

    fname = tstr
    # Save image with its label
    img.save(os.path.join(output_dir, f"{i}_{fname}_synthetic_at.png"))



for i in range(100):
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


for i in range(200):
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


for i in range(50):
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


