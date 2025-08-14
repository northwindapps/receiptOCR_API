import cv2
import numpy as np

def isolate_receipt_complete(image_path):
    """
    Complete pipeline to isolate receipt from background
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Threshold to separate receipt from background
    # Since receipt is lighter than background
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # Step 2: Clean up the binary image
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Step 3: Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Find the largest contour (should be the receipt)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small margin
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2*margin)
        h = min(img.shape[0] - y, h + 2*margin)
        
        # Crop the receipt
        receipt_crop = img[y:y+h, x:x+w]
        
        # Optional: Create a clean white background version
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        mask_crop = mask[y:y+h, x:x+w]
        
        # Apply mask to get receipt with white background
        receipt_clean = np.ones_like(receipt_crop) * 255
        receipt_clean[mask_crop > 0] = receipt_crop[mask_crop > 0]
        
        return receipt_crop, receipt_clean, (x, y, w, h)
    
    return img, img, None

# Alternative method using adaptive threshold
def isolate_receipt_adaptive(image_path):
    """
    Use adaptive thresholding for varying lighting
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 21, 10)
    
    # Invert if needed (we want receipt to be white)
    if np.mean(thresh) < 127:
        thresh = 255 - thresh
    
    # Fill holes and clean up
    kernel = np.ones((10,10), np.uint8)
    filled = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Create mask from contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        
        # Crop
        cropped = img[y:y+h, x:x+w]
        
        return cropped, mask
    
    return img, None

# Use the functions
image_path = r'C:\Users\ABC\Documents\receiptYOLOProject\test17.jpg'

# Try method 1
crop1, clean1, bbox = isolate_receipt_complete(image_path)
cv2.imwrite('receipt_isolated_method1.jpg', crop1)
cv2.imwrite('receipt_clean_method1.jpg', clean1)

# Try method 2  
crop2, mask2 = isolate_receipt_adaptive(image_path)
cv2.imwrite('receipt_isolated_method2.jpg', crop2)

# Show the results
cv2.imshow('Original', cv2.resize(cv2.imread(image_path), (400, 600)))
cv2.imshow('Isolated Receipt', cv2.resize(crop1, (400, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()