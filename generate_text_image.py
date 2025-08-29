from PIL import Image, ImageDraw, ImageFont
import random

def generate_text_image(text, font_path="arial.ttf", size=32):
    # Create a blank white image
    img = Image.new("L", (200, size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Pick a font
    font = ImageFont.truetype(font_path, size-4)
    
    # Draw text
    draw.text((5, 0), text, font=font, fill=0)
    
    # Random distortions (optional)
    # img = img.rotate(random.randint(-2, 2), expand=True, fillcolor=255)
    
    return img

# Example
img = generate_text_image("123.45")
img.show()
