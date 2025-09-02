from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

def generate_receipt():
    # blank background
    img = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(img)

    # fonts (need to have some TTF fonts installed)
    font = ImageFont.truetype("arial.ttf", 28)

    y = 50
    items = ["Croissant", "Cappuccino", "Sandwich", "Tea"]
    total = 0

    for i in range(3):
        item = random.choice(items)
        price = round(random.uniform(2, 10), 2)
        total += price
        draw.text((50, y), f"{item}", font=font, fill="black")
        draw.text((400, y), f"${price:.2f}", font=font, fill="black")
        y += 50

    draw.text((50, y+30), "Total", font=font, fill="black")
    draw.text((400, y+30), f"${total:.2f}", font=font, fill="black")

    return img

img = generate_receipt()
img.show()
