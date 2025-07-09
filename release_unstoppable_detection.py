from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from typing import Tuple
import time


# unstoppable_path = Path("screenshots\\screenshot_20250707_204245_848440.jpg")
unstoppable_path = Path("screenshots\\screenshot_20250707_204156_074363.jpg")

image_crop_triangle = {"x1": 550, "y1": 765, "x2": 1350, "y2": 785}

image_crop_green= {"x1": 550, "y1": 750, "x2": 1350, "y2": 850}

image = cv2.cvtColor(cv2.imread(str(unstoppable_path)), cv2.COLOR_BGR2RGB)

green_start = time.time()
image_cropped_green = image[image_crop_green["y1"]:image_crop_green["y2"], image_crop_green["x1"]:image_crop_green["x2"]]

# Create a mask where the green channel is greater than 230
green_channel = image_cropped_green[:, :, 1]
red_channel = image_cropped_green[:, :, 0]
blue_channel = image_cropped_green[:, :, 2]
green_mask = (green_channel > 230) & (blue_channel < 70) & (red_channel < 130)

# Create a thresholded image: white where green > 230, black elsewhere
image_cropped_thresholded = np.zeros_like(green_channel, dtype=np.uint8)
image_cropped_thresholded[green_mask] = 255

# Find coordinates of white pixels (value 255) in the thresholded image
white_pixels = np.column_stack(np.where(image_cropped_thresholded == 255))

if white_pixels.size > 0:
    # Each row is (y, x). To get leftmost, find min x; rightmost, max x.
    leftmost_idx = np.argmin(white_pixels[:, 1])
    rightmost_idx = np.argmax(white_pixels[:, 1])
    leftmost_pixel = tuple(white_pixels[leftmost_idx][::-1])  # (x, y)
    rightmost_pixel = tuple(white_pixels[rightmost_idx][::-1])  # (x, y)
    print(f"Leftmost white pixel: {leftmost_pixel}")
    print(f"Rightmost white pixel: {rightmost_pixel}")
else:
    print("No white pixels found.")

print(f"Green time: {time.time() - green_start}")

cv2.imshow("", image_cropped_thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()

triangle_start = time.time()
image_cropped_triangle = image[image_crop_triangle["y1"]:image_crop_triangle["y2"], image_crop_triangle["x1"]:image_crop_triangle["x2"]]
image_cropped_gray = cv2.cvtColor(image_cropped_triangle, cv2.COLOR_BGR2GRAY)

image_cropped_thresholded = cv2.threshold(image_cropped_gray, 230, 255, cv2.THRESH_BINARY)[1]

# Detect all triangles in the thresholded image and return their x-coordinates

# Find contours in the thresholded image
contours, _ = cv2.findContours(image_cropped_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume only one triangle exists, so take the largest contour
cnt = max(contours, key=cv2.contourArea)

# Approximate the contour to a polygon
epsilon = 0.04 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

# Draw the triangle for visualization
image_to_draw = cv2.cvtColor(image_cropped_thresholded, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_to_draw, [approx], -1, (0, 255, 0), 2)

# Calculate the centroid of the triangle

M = cv2.moments(approx)
if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    print("Triangle centroid x-coordinate:", cx)
else:
    print("Centroid calculation failed (zero area).")

print(f"Triangle time: {time.time() - triangle_start}")

in_area = cx > leftmost_pixel[0] and cx < rightmost_pixel[0]

print(f"In area: {in_area}")

# Show the image with drawn contours
cv2.imshow("Approximated Contours", image_to_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("", image_cropped_thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()







