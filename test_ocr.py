import cv2
import pytesseract
import numpy as np
from pathlib import Path
import os

# Set the path to tesseract executable
# Change this path according to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy while preserving details
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding instead of global Otsu
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply very light denoising to preserve details
    # gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # do a morphological closing to fill in small holes
    # kernel = np.ones((3,3), np.uint8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    #apply a median blur to reduce noise
    # gray = cv2.medianBlur(gray, 3)

    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return gray

def perform_ocr(image_path):
    """
    Perform OCR on the given image with high accuracy settings
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Preprocess nthe image
    processed_image = preprocess_image(image)
    processed_image = processed_image[:,15:]

    cv2.imshow("Preprocessed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Perform OCR with optimized settings for all ASCII characters
    custom_config = r'--oem 3 --psm 6'  # Removed character whitelist to allow all characters
    text = pytesseract.image_to_string(processed_image, config=custom_config)
    
    return text.strip()

def main():
    # Path to the OCR test images
    ocr_test_dir = Path("img/ocr_test")
    
    if not ocr_test_dir.exists():
        print(f"Error: Directory {ocr_test_dir} does not exist")
        return
    
    # Process all images in the directory
    for image_path in ocr_test_dir.glob("*"):
        if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            print(f"\nProcessing: {image_path.name}")
            print("-" * 50)
            try:
                text = perform_ocr(image_path)
                print("OCR Result:")
                print(text)
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")

if __name__ == "__main__":
    main() 