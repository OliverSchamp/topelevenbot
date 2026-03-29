import cv2
import time
from pathlib import Path

def find_target_fast(screen_path, template_path):
    # 1. Read images in Grayscale (IMREAD_GRAYSCALE is a massive speed boost)
    screen = cv2.imread(screen_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if screen is None or template is None:
        print("Error: Could not load images.")
        return None

    # Get template dimensions to calculate the center later
    h, w = template.shape

    # Start timer
    start_time = time.perf_counter()

    # 2. Perform the Template Match
    # TM_CCOEFF_NORMED is highly robust to lighting changes while remaining fast
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)

    # 3. Find the exact pixel coordinate with the highest match confidence
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 4. Calculate the center of the target
    # max_loc gives the top-left corner of the matched area
    center_x = max_loc[0] + (w // 2)
    center_y = max_loc[1] + (h // 2)

    detection_time = (time.perf_counter() - start_time) * 1000

    print(f"Target found at (X: {center_x}, Y: {center_y})")
    print(f"Confidence: {max_val:.2f} (1.0 is a perfect match)")
    print(f"Detection Time: {detection_time:.3f} ms")

    return (center_x, center_y), max_val

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    parent = Path("/mnt/SF_NAS/Oliver")
    # Replace these with your actual file names
    screen_file = parent / "1773777133.7.jpg"
    template_file = Path("/mnt/SF_NAS/Oliver/tmplt.jpg")
    
    find_target_fast(screen_file, template_file)