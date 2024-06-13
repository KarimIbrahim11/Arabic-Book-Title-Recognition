import numpy as np
import cv2

def enhance_contrast(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    # Convert back to BGR
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

def swap_columns(data, col1_idx, col2_idx):
    for row in data:
        row[col1_idx], row[col2_idx] = row[col2_idx], row[col1_idx]
def OCR_preprocess_image(image):
    # Convert to grayscale
    enhanced = enhance_contrast(image)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding
    # # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # Remove noise with morphological operations (optional)
    # kernel = np.ones((1, 1), np.uint8)
    # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    #
    # # Resize the image (optional, depends on the OCR tool)
    # height, width = morph.shape
    # new_height, new_width = height * 2, width * 2
    # resized = cv2.resize(morph, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return blurred
