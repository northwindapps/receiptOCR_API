import cv2
import numpy as np
import imutils
import pytesseract
from matplotlib import pyplot as plt


def preprocessing(img):
  increase = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
  gray = cv2.cvtColor(increase, cv2.COLOR_BGR2GRAY)
  value, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  return otsu

def show_img(img):
  fig = plt.gcf()
  fig.set_size_inches(16, 8)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test30.jpg'
image = cv2.imread(image_path)
processed_plate = preprocessing(image)
show_img(processed_plate)