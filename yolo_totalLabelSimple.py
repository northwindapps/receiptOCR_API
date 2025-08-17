import cv2
import pytesseract
from ultralytics import YOLO
import os,re
import easyocr
import datetime

def rotate(img,v):
    # Get image dimensions
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix (positive angle = counter-clockwise)
    M = cv2.getRotationMatrix2D(center, v, 1.0)  # angle=2 degrees, scale=1.0

    # Apply rotation
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    return rotated
def safe_filename(text):
    # Replace forbidden characters with underscore
    return re.sub(r'[\\/*?:"<>|]', "_", text)

# Load models
totalLabel_model = YOLO('totalLabel_best.pt')         

chunk_model = YOLO('text_chunk_epoch40_best.pt')

# Get current datetime string (YYYYMMDD_HHMMSS format)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Image path
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test26.jpg'
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
# alpha_values = [-3.0,-1.0,1.5,3.0,5.0]  # contrast
# beta_values = [-80,-50,0,50,80,160]  # brightness
alpha_values = [-2.0,-1.0,1.0,2.0]  # contrast
beta_values = [100,80,70,50,25,0,-25,-50,-80,-100]
cont_area_values = [0,55]
rotate_degrees = [0]
scales = [1.5,2.5,3.0]
# Run layout detection
totalLabel_results = totalLabel_model(source=sharpened, conf=0.50, save=True, show=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Change gpu=True if you have a GPU and want to use it
skip_box = False
for idx, totalLabel_result in enumerate(totalLabel_results):
    for jdx, totalLabel_box in enumerate(totalLabel_result.boxes):
        x_min, y_min, x_max, y_max = map(int, totalLabel_box.xyxy[0].tolist())

        alpha_stop = False
        stop_early = False  # flag to exit all loop
        target_conf = 0.9
        best_text = ""
        best_conf = 0
        prev = ""
        count = 0
        count_50 = 0

        # Loop through margin adjustments
        for margin_adjust in [-3,0,3]:#[-8,-5,-1, 0, 1,5,8]:
            y_min_adj = max(0, y_min - margin_adjust)
            y_max_adj = y_max
            if stop_early:
                break

            # Loop through all alpha/beta combinations
            for cont_value in cont_area_values:
                alpha_stop = False
                layout_crop = sharpened[int(y_min_adj):int(y_max_adj), int(x_max+0):img_width]    
                for av in alpha_values:
                    print(f"alpha{av}")
                    stop_early = False
                    if alpha_stop:
                        break
                    for scale in scales:
                        if stop_early:
                             break 
                        for bv in beta_values:
                            if stop_early:
                             break 
                            # rotated = rotate(layout_crop,degree)
                            # contrast = cv2.convertScaleAbs(rotated, alpha=av, beta=bv)
                            contrast = layout_crop
                            processed_img = cv2.resize(contrast, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                            g = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                            
                            thresh = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)

                            margin = 20
                            if len(thresh.shape) == 2:  # grayscale
                                chunk_crop_with_bg = cv2.copyMakeBorder(
                                    thresh, margin, margin, margin, margin,
                                    cv2.BORDER_CONSTANT, value=255
                                )
                            else:  # color (3 channels)
                                chunk_crop_with_bg = cv2.copyMakeBorder(
                                    thresh, margin, margin, margin, margin,
                                    cv2.BORDER_CONSTANT, value=[255,255,255]
                                )

                            
                            thresh = chunk_crop_with_bg
                            

                            # Remove small blobs
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                if cv2.contourArea(cnt) < cont_value:
                                    cv2.drawContours(thresh, [cnt], -1, 0, -1)

                            # Run Tesseract
                            data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)
                            avg_conf = sum(int(c) for c in data['conf'] if int(c) >= 0) / max(1, len([c for c in data['conf'] if int(c) >= 0]))
                            avg_conf /= 100

                            text = pytesseract.image_to_string(thresh, lang='eng', config="--psm 7 -c tessedit_char_whitelist=0123456789:.$").strip()
                            # text = pytesseract.image_to_string(thresh, lang='eng').strip()


                            if avg_conf > best_conf:
                                best_conf = avg_conf
                                best_text = text
                            if avg_conf < 0.5:
                                # want to move next alpha
                                stop_early = True
                        
                        if stop_early:
                            break 
                        if best_conf >= 0.70 and prev == best_text:
                            count += 1
                        else:
                            prev = best_text
                            count = 0
                        if (best_conf < 0.70 and best_conf > 0.50) and prev == best_text:
                            count_50 += 1
                        else:
                            prev = best_text
                            count_50 = 0
                        # After trying all alpha/beta/margin combinations
                        if best_conf >= target_conf:
                            # print(f"Final OCR Text: {best_text}, Confidence: {best_conf:.2f}")
                            # print(f"Box: [{x_min}, {y_min}, {x_max}, {y_max}]")
                            # # Save the cropped image
                            # crop_filename = os.path.join(save_dir, f"90_crop_{idx}_{jdx}_alpha{av}_beta{bv}_pyttext_{safe_filename(best_text)}_box_[{x_min}_{y_min}_{x_max}_{y_max}]_ts_{timestamp}.png")
                            # cv2.imwrite(crop_filename, thresh)
                            # print(f"Saved crop: {crop_filename}")
                            # print(f"alpha:beta:contour: [{av}, {bv}, {cont_value}]")
                            stop_early = True
                            alpha_stop = True
                            prev = ''
                            count = 0
                            count_50 = 0
                            print("easyOCR")
                            # Convert BGR to RGB for EasyOCR
                            chunk_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

                            # OCR with EasyOCR
                            results = reader.readtext(chunk_rgb, detail=1)
                            clean_text = ''
                            conf = ''
                            for bbox, text, confidence in results:
                                # print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")
                                conf = confidence
                                clean_text = text.replace("-", ".")
                                print(clean_text)
                                if any(char.isdigit() for char in clean_text) and not clean_text.endswith(".") and not clean_text.startswith("."):
                                    if confidence >= 0.9:
                                        stop_early = True
                                        print(f"Chunk Text: {clean_text}")
                                        print(f"Confidence: {confidence:.2f}")
                                        print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                        print("Contains numbers")
                                        break
                                    if confidence >= 0.7 and "$" in clean_text:
                                        stop_early = True
                                        print(f"Chunk Text: {clean_text}")
                                        print(f"Confidence: {confidence:.2f}")
                                        print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                        print("Contains $")
                                        break
                            # Save the cropped image
                            crop_filename = os.path.join(save_dir, f"90_crop_{idx}_{jdx}_alpha{av}_beta{bv}_pyttext_{safe_filename(best_text)}_easyOCR_text_{safe_filename(clean_text)}_conf_{conf:.2f}_box_[{x_min}_{y_min}_{x_max}_{y_max}]_ts_{timestamp}.png")
                            cv2.imwrite(crop_filename, thresh)
                            print(f"Saved crop: {crop_filename}")
                            print(f"alpha:beta:contour: [{av}, {bv}, {cont_value}]")
                            break
                        else:
                            print(f"No high-confidence result found, best was: {best_text} ({best_conf:.2f})")
                            # stop_early = True
                            if any(char.isdigit() for char in best_text) and (count > 2 or count_50 > 3):
                                stop_early = True
                                alpha_stop = True
                                prev = ''
                                count = 0
                                count_50 = 0
                                
                                print("easyOCR")
                                # Convert BGR to RGB for EasyOCR
                                chunk_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

                                # OCR with EasyOCR
                                results = reader.readtext(chunk_rgb, detail=1)
                                clean_text = ''
                                conf = ''
                                for bbox, text, confidence in results:
                                    # print(f"Chunk BBox: [{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]")
                                    conf = confidence
                                    clean_text = text.replace("-", ".")
                                    print(clean_text)
                                    if any(char.isdigit() for char in clean_text) and not clean_text.endswith(".") and not clean_text.startswith("."):
                                        if confidence >= 0.9:
                                            stop_early = True
                                            print(f"Chunk Text: {clean_text}")
                                            print(f"Confidence: {confidence:.2f}")
                                            print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                            print("Contains numbers")
                                            break
                                        if confidence >= 0.7 and "$" in clean_text:
                                            stop_early = True
                                            print(f"Chunk Text: {clean_text}")
                                            print(f"Confidence: {confidence:.2f}")
                                            print(f"Chunk BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                            print("Contains $")
                                            break
                                # Save the cropped image
                                crop_filename = os.path.join(save_dir, f"70_crop_{idx}_{jdx}_alpha{av}_beta{bv}_pyttext_{safe_filename(best_text)}_easyOCR_text_{safe_filename(clean_text)}_conf_{conf:.2f}_box_[{x_min}_{y_min}_{x_max}_{y_max}]_ts_{timestamp}.png")
                                cv2.imwrite(crop_filename, thresh)
                                print(f"Saved crop: {crop_filename}")
                                print(f"alpha:beta:contour: [{av}, {bv}, {cont_value}]")
                                break

                        if best_conf > 0.90 and any(char.isdigit() for char in best_text) and "." in best_text:
                            stop_early = True
                            alpha_stop = True
                            print("Contains numbers")
                            break
                        # elif best_conf >= 0.75 and any(char.isdigit() for char in best_text)  and "$" in best_text and "." in best_text:
                        #     stop_early = True
                        #     print(f"alpha:{alpha}")
                        #     print(f"beta:{beta}")
                        #     print("Contains numbers")
                        #     break
                        elif count < -5:
                            stop_early = True
                            prev = ''
                            count = 0
                            break
                        else:
                            break
