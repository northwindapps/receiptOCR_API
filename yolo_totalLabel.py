import cv2
import pytesseract
from ultralytics import YOLO
import os
import easyocr

# Load models
totalLabel_model = YOLO('totalLabel_best.pt')         

chunk_model = YOLO('text_chunk_epoch40_best.pt')

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test9.jpg'
image = cv2.imread(image_path)


# Parameters to tweak sharpness
strength = 1.5  # How much to boost edges (1.0 = no change)
blur_size = (0, 0)  # Let OpenCV figure size from sigma
sigma = 3  # How much to blur before subtracting

# Unsharp masking
blurred = cv2.GaussianBlur(image, blur_size, sigma)
sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

# Contrast
alpha = 5.0  #1.5,2,3,4,5 contrast factor
beta = 80#-100,-80,50,0,50,80,100    # brightness offset (tweak if needed)

bright_contrast_image = cv2.convertScaleAbs(src=sharpened,alpha=alpha,beta=beta)
# bright_contrast_image = image

# Get image width
img_height, img_width = image.shape[:2]

# Create folder to save crops
save_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\crops"
os.makedirs(save_dir, exist_ok=True)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Run layout detection
totalLabel_results = totalLabel_model(source=bright_contrast_image, conf=0.50, save=True, show=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have a GPU and want to use it

for idx,totalLabel_result in enumerate(totalLabel_results):
    for jdx,totalLabel_box in enumerate(totalLabel_result.boxes):
        # Layout region coords
        x_min, y_min, x_max, y_max = map(int, totalLabel_box.xyxy[0].tolist())

        target_conf = 0.8
        best_text = ""
        best_conf = 0

        for margin_adjust in [-5,-3,-2,-1, 0, 1, 2,3,5]:  # pixels to add/remove
            # Adjust crop vertically
            y_min_adj = max(0, y_min - margin_adjust)
            y_max_adj = y_max #min(img_height, y_max + margin_adjust)
            
            layout_crop = bright_contrast_image[int(y_min_adj):int(y_max_adj), int(x_max+0):img_width]
            
            scale_factor = 3
            resized_chunk = cv2.resize(layout_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


            margin = 20

            # Optional: make white background if original is grayscale
            if len(resized_chunk.shape) == 2:  # grayscale
                chunk_crop_with_bg = cv2.copyMakeBorder(
                    resized_chunk, margin, margin, margin, margin,
                    cv2.BORDER_CONSTANT, value=255
                )
            else:  # color image
                chunk_crop_with_bg = cv2.copyMakeBorder(
                    resized_chunk, margin, margin, margin, margin,
                    cv2.BORDER_CONSTANT, value=[255, 255, 255]
                )


            # OCR processing
            gray = cv2.cvtColor(chunk_crop_with_bg, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 15:  # threshold tiny blobs
                    cv2.drawContours(thresh, [cnt], -1, 0, -1)  # fill with black
                
            data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
            avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
            avg_conf /= 100

            text = pytesseract.image_to_string(thresh, lang='eng', config="--psm 7 -c tessedit_char_whitelist=0123456789:.").strip()
            
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_text = text

            # if avg_conf >= target_conf:
                # break  # stop early if we hit the goal

        # After loop
        if best_conf >= target_conf:
            print(f"Final OCR Text: {best_text}, Confidence: {best_conf:.2f}")
            print(f"Box: [{x_min}, {y_min}, {x_max}, {y_max}]")
        else:
            print(f"No high-confidence result found, best was: {best_text} ({best_conf:.2f})")

        # Save the cropped image
        crop_filename = os.path.join(save_dir, f"totalLabel_crop_{idx}_{jdx}.png")
        cv2.imwrite(crop_filename, thresh)
        print(f"Saved crop: {crop_filename}")

        # Convert BGR to RGB for EasyOCR
        chunk_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

        # OCR with EasyOCR
        results = reader.readtext(thresh, detail=1)

        for bbox, text, confidence in results:
            if confidence > 0.8:
                print(f"Chunk Text: {text}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                # print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")

                print("=" * 40)
















        # # Crop from totalLabel box x_min to the right edge
        # layout_crop = bright_contrast_image[int(y_min)+2:int(y_max)-0, int(x_max)+0:img_width]

        # scale_factor = 3
        # resized_chunk = cv2.resize(layout_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


        # margin = 20

        # # Optional: make white background if original is grayscale
        # if len(resized_chunk.shape) == 2:  # grayscale
        #     chunk_crop_with_bg = cv2.copyMakeBorder(
        #         resized_chunk, margin, margin, margin, margin,
        #         cv2.BORDER_CONSTANT, value=255
        #     )
        # else:  # color image
        #     chunk_crop_with_bg = cv2.copyMakeBorder(
        #         resized_chunk, margin, margin, margin, margin,
        #         cv2.BORDER_CONSTANT, value=[255, 255, 255]
        #     )

    
        #  # Preprocess for OCR
        # gray = cv2.cvtColor(chunk_crop_with_bg, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     if cv2.contourArea(cnt) < 10:  # threshold tiny blobs
        #         cv2.drawContours(thresh, [cnt], -1, 0, -1)  # fill with black

        # # Save the cropped image
        # crop_filename = os.path.join(save_dir, f"totalLabel_crop_{idx}_{jdx}.png")
        # cv2.imwrite(crop_filename, thresh)
        # print(f"Saved crop: {crop_filename}")
        
        # # OCR
        # # text = pytesseract.image_to_string(thresh, lang='eng').strip()
        # config = "--psm 7 -c tessedit_char_whitelist=0123456789:."
        # text = pytesseract.image_to_string(thresh, lang='eng', config=config)

        # # text = pytesseract.image_to_string(thresh, lang='eng').strip()
        # data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
        # avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
        # avg_conf /= 100
        # if text != '' and avg_conf > 0.5:
        #     print(f"Layout Class: {int(totalLabel_box.cls[0])}, Conf: {float(totalLabel_box.conf[0]):.2f}")
        #     print(f"Average OCR Confidence: {avg_conf:.2f}")
        #     # print(f"Chunk Class: {int(chunk_box.cls[0])}, Conf: {float(chunk_box.conf[0]):.2f}")
        #     print(f"Chunk Text: {text}")
        #     print(f"Chunk BBox (original image coords): [{x_min}, {y_min}, {x_max}, {y_max}]")
        #     print("="*40)

        # # Convert BGR to RGB for EasyOCR
        # chunk_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

        # # OCR with EasyOCR
        # results = reader.readtext(thresh, detail=1)

        # for bbox, text, confidence in results:
        #     print(f"Chunk Text: {text}")
        #     print(f"Confidence: {confidence:.2f}")
        #     print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
        #     # print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")

        #     print("=" * 40)
