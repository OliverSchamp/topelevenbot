from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from typing import Tuple
import time

# def _check_every_5th_pixel(img: Image.Image, target_rgb_low: Tuple[int, int, int], target_rgb_high: Tuple[int, int, int]) -> bool:
#         arr = np.array(img)
#         h, w, _ = arr.shape
#         for y in range(0, h, 5):
#             for x in range(0, w, 5):
#                 if target_rgb_low[0] <= arr[y, x][0] <= target_rgb_high[0] and target_rgb_low[1] <= arr[y, x][1] <= target_rgb_high[1] and target_rgb_low[2] <= arr[y, x][2] <= target_rgb_high[2]:
#                     return True
#         return False

# unstoppable_path = Path("screenshots\\screenshot_20250707_204147_569962.jpg")

unstoppable_path = Path("screenshots\\screenshot_20250707_204244_848268.jpg")


image_crop = {"x1": 250, "y1": 120, "x2": 1650, "y2": 600}

image = cv2.cvtColor(cv2.imread(str(unstoppable_path)), cv2.COLOR_BGR2RGB)

image_cropped = image[image_crop["y1"]:image_crop["y2"], image_crop["x1"]:image_crop["x2"]]

# start = time.time()
# # print(_check_every_5th_pixel(image_cropped, (40, 250, 5), (50, 255, 15)))
# print(f"Time taken: {time.time() - start} seconds")

# If a matching pixel is found, display the image with the pixel marked
def find_matching_pixel(img: np.ndarray, target_rgb_low: Tuple[int, int, int], target_rgb_high: Tuple[int, int, int]):
    h, w, _ = img.shape
    for y in range(0, h, 5):
        for x in range(0, w, 5):
            pixel = img[y, x]
            if (target_rgb_low[0] <= pixel[0] <= target_rgb_high[0] and
                target_rgb_low[1] <= pixel[1] <= target_rgb_high[1] and
                target_rgb_low[2] <= pixel[2] <= target_rgb_high[2]):
                return (x, y)
    return None
start = time.time()
match = find_matching_pixel(image_cropped, (40, 250, 5), (50, 255, 15))
print(f"Time taken: {time.time() - start} seconds")
if match is not None:
    # Draw a red circle at the found pixel
    display_img = image_cropped.copy()
    cv2.circle(display_img, match, 8, (0, 0, 255), 2)
    cv2.imshow("Found Pixel", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







