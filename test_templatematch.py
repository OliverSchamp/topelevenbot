import cv2
from pathlib import Path
import time

template_path = Path("img/test/greens_ads_button.jpg")
image_path = Path("img/test/watchads.png")

template = cv2.imread(str(template_path))
image = cv2.imread(str(image_path))

result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

w, h = template.shape[1], template.shape[0]
top_left_x, top_left_y = max_loc

center_x = top_left_x + w//2
center_y = top_left_y + h//2
print(f"Found {max_loc} with confidence: {max_val:.2%}")