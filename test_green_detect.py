import cv2
import numpy as np

def is_text_green(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found."

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    # Hue: ~35-85, Saturation: 50-255, Value: 50-255
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create a mask that only includes green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Count how many green pixels were found
    green_pixel_count = cv2.countNonZero(mask)

    # If more than 100 pixels are green, return True
    # (Threshold adjusted for the size of the clock digits)
    return green_pixel_count > 100

# Example usage:
print(is_text_green('/mnt/SF_NAS/Oliver/red.jpg')) # Should return True
# print(is_text_green('00_06.jpg')) # Should return False