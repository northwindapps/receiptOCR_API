import cv2
import pytesseract
from ultralytics import YOLO
import os
import easyocr

def rotate(img,v):
    # Get image dimensions
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix (positive angle = counter-clockwise)
    M = cv2.getRotationMatrix2D(center, v, 1.0)  # angle=2 degrees, scale=1.0

    # Apply rotation
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    return rotated

# Load models
totalLabel_model = YOLO('totalLabel_best.pt')         

chunk_model = YOLO('text_chunk_epoch40_best.pt')

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test19.jpg'
image = cv2.imread(image_path)


sharpened = image
# Get image width
img_height, img_width = image.shape[:2]

# Create folder to save crops
save_dir = r"C:\Users\ABC\Documents\receiptYOLOProject\crops"
os.makedirs(save_dir, exist_ok=True)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have a GPU and want to use it

 # Define contrast & brightness variations to try
alpha_values = [-5.0,-3.0,-1.0,1.0]  #1.0 contrast
# beta_values = [-80,-50,0,50,80,160]  # brightness
# alpha_values = [1.5]  # contrast
# beta_values = [-80,-50,0,50,80,160]
beta_values = [150,120,80,50,0,-50,-80]
cont_area_values = [55]
rotate_values = [-2.0,-1.5,-1.0,0,1.0,1.5,2.0]
# Run layout detection
totalLabel_results = totalLabel_model(source=sharpened, conf=0.50, save=True, show=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have a GPU and want to use it
print(image_path)
for idx, totalLabel_result in enumerate(totalLabel_results):
    for jdx, totalLabel_box in enumerate(totalLabel_result.boxes):
        x_min, y_min, x_max, y_max = map(int, totalLabel_box.xyxy[0].tolist())

        stop_early = False  # flag to exit all loop
        target_conf = 0.9
        best_text = ""
        best_conf = 0
        prev = ""
        count = 0

        # Loop through margin adjustments
        for margin_adjust in [0]:
            y_min_adj = max(0, y_min - margin_adjust)
            y_max_adj = y_max
            if stop_early:
                break

            # Loop through all alpha/beta combinations
            for alpha in alpha_values:
                for beta in beta_values:
                    for cont_value in cont_area_values:
                        print(f"alpha:beta:contour: [{alpha}, {beta}, {cont_value}]")
                        # Apply brightness & contrast adjustment to original sharpened image
                        bc_image = cv2.convertScaleAbs(src=sharpened, alpha=alpha, beta=beta)
                        layout_crop = bc_image[int(y_min_adj):int(y_max_adj), int(x_max+0):img_width]
                        for rv in rotate_values:
                            rotated = rotate(layout_crop,rv)
                            scale_factor = 3
                            resized_chunk = cv2.resize(rotated, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

                            # Parameters to tweak sharpness
                            strength = 1.5  # How much to boost edges (1.0 = no change)
                            blur_size = (0, 0)  # Let OpenCV figure size from sigma
                            sigma = 3  # How much to blur before subtracting

                            # Unsharp masking
                            blurred = cv2.GaussianBlur(resized_chunk, blur_size, sigma)
                            resized_chunk = cv2.addWeighted(resized_chunk, 1 + strength, blurred, -strength, 0)

                            margin = 20
                            if len(resized_chunk.shape) == 2:
                                chunk_crop_with_bg = cv2.copyMakeBorder(
                                    resized_chunk, margin, margin, margin, margin,
                                    cv2.BORDER_CONSTANT, value=255
                                )
                            else:
                                chunk_crop_with_bg = cv2.copyMakeBorder(
                                    resized_chunk, margin, margin, margin, margin,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255]
                                )

                            gray = cv2.cvtColor(chunk_crop_with_bg, cv2.COLOR_BGR2GRAY)
                            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                            # Remove small blobs
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                if cv2.contourArea(cnt) < cont_value:
                                    cv2.drawContours(thresh, [cnt], -1, 0, -1)

                            # Save the cropped image
                            crop_filename = os.path.join(save_dir, f"totalLabel_crop_{idx}_{jdx}.png")
                            cv2.imwrite(crop_filename, thresh)
                            print(f"Saved crop: {crop_filename}")

                            # Run Tesseract
                            data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
                            avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
                            avg_conf /= 100

                            text = pytesseract.image_to_string(thresh, lang='eng', config="--psm 7 -c tessedit_char_whitelist=0123456789:.$").strip()
                            # text = pytesseract.image_to_string(thresh, lang='eng').strip()


                            if avg_conf > best_conf:
                                best_conf = avg_conf
                                best_text = text
                            if avg_conf < 0.6:
                                break
                        
                        if stop_early:
                                    break 
                        if best_conf >= 0.75 and prev == best_text:
                            count += 1
                        else:
                            prev = best_text
                            count = 0
                        # After trying all alpha/beta/margin combinations
                        if best_conf >= target_conf:
                            print(f"Final OCR Text: {best_text}, Confidence: {best_conf:.2f}")
                            print(f"Box: [{x_min}, {y_min}, {x_max}, {y_max}]")
                        else:
                            print(f"No high-confidence result found, best was: {best_text} ({best_conf:.2f})")
                            if best_conf == 0.00:
                                count -= 1

                        if best_conf > 0.90 and any(char.isdigit() for char in best_text) and "." in best_text:
                            stop_early = True
                            print(f"alpha:{alpha}")
                            print(f"beta:{beta}")
                            print("Contains numbers")
                            break
                        # elif best_conf >= 0.75 and any(char.isdigit() for char in best_text)  and "$" in best_text and "." in best_text:
                        #     stop_early = True
                        #     print(f"alpha:{alpha}")
                        #     print(f"beta:{beta}")
                        #     print("Contains numbers")
                        #     break
                        elif best_conf >= 0.75 and any(char.isdigit() for char in best_text) and count > 3 and "." in best_text:
                            stop_early = True
                            print(f"alpha:{alpha}")
                            print(f"beta:{beta}")
                            prev = ''
                            count = 0
                            break
                        elif count < -5:
                            stop_early = True
                            prev = ''
                            count = 0
                            break
                        else:
                            print("easyOCR")
                            # Convert BGR to RGB for EasyOCR
                            chunk_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

                            # OCR with EasyOCR
                            results = reader.readtext(chunk_rgb, detail=1)

                            for bbox, text, confidence in results:
                                if stop_early:
                                    break 
                                
                                # print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")

                                clean_text = text.replace("-", ".")
                                if any(char.isdigit() for char in clean_text) and not clean_text.endswith(".") and not clean_text.startswith("."):
                                    if confidence >= 0.9 and "." in clean_text:
                                        stop_early = True
                                        print(f"Chunk Text: {clean_text}")
                                        print(f"Confidence: {confidence:.2f}")
                                        print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                        print(f"alpha:{alpha}")
                                        print(f"beta:{beta}")
                                        print("Contains numbers")
                                        break
                                    if confidence >= 0.7:
                                        stop_early = True
                                        print(f"Chunk Text: {clean_text}")
                                        print(f"Confidence: {confidence:.2f}")
                                        print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                        print(f"alpha:{alpha}")
                                        print(f"beta:{beta}")
                                        print("Contains $")
                                        break

                            else:
                                print("No numbers found")

                            