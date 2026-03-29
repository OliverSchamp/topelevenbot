import subprocess
import io
from PIL import Image
import time
import numpy as np
from pathlib import Path

def capture_waydroid_optimized(x, y, w, h):
    # -t ppm is the key for speed on Pi 5
    region = f"{x},{y} {w}x{h}"
    command = ['grim', '-t', 'ppm', '-g', region, '-']
    
    start_capture = time.time()
    result = subprocess.run(command, capture_output=True, check=True)
    
    # Fast parsing with PIL
    img = Image.open(io.BytesIO(result.stdout))
    img.load() # Force memory load
    
    print(f"Capture + Load: {time.time() - start_capture:.4f}s")
    return img

def capture_waydroid_optimized_np(x, y, w, h):
    region = f"{x},{y} {w}x{h}"
    command = ['grim', '-t', 'ppm', '-g', region, '-']
    start_capture = time.time()
    result = subprocess.run(command, capture_output=True, check=True)
    
    data_start = result.stdout.find(b'\n', result.stdout.find(b'\n', result.stdout.find(b'\n') + 1) + 1) + 1
    
    flat_array = np.frombuffer(result.stdout[data_start:], dtype=np.uint8)
    img = flat_array.reshape((height, width, 3))
    print(f"Capture + Load: {time.time() - start_capture:.4f}s")
    return img


width = 960
height = 540
SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080
taskbar_offset_px = 18

def take_screenshot_wh(rel_x: int = 0, rel_y: int = 0, rel_width: int = width, rel_height: int = height) -> np.ndarray:
    roi_base_x = (SCREEN_WIDTH - width) // 2
    roi_base_y = (SCREEN_HEIGHT - height) // 2 + taskbar_offset_px
    abs_x = roi_base_x + rel_x
    abs_y = roi_base_y + rel_y
    region = f"{abs_x},{abs_y} {rel_width}x{rel_height}"
    command = ['grim', '-t', 'ppm', '-g', region, '-']
    result = subprocess.run(command, capture_output=True, check=True)
    data_start = result.stdout.find(b'\n', result.stdout.find(b'\n', result.stdout.find(b'\n') + 1) + 1) + 1
    flat_array = np.frombuffer(result.stdout[data_start:], dtype=np.uint8)
    img = flat_array.reshape((rel_height, rel_width, 3))
    
    return img

def take_screenshot(rel_x: int = 0, rel_y: int = 0, rel_x2: int = width, rel_y2: int = height) -> np.ndarray:
    rel_width = rel_x2 - rel_x
    rel_height = rel_y2 - rel_y
    roi_base_x = (SCREEN_WIDTH - width) // 2
    roi_base_y = (SCREEN_HEIGHT - height) // 2 + taskbar_offset_px
    abs_x = roi_base_x + rel_x
    abs_y = roi_base_y + rel_y
    region = f"{abs_x},{abs_y} {rel_width}x{rel_height}"
    command = ['grim', '-t', 'ppm', '-g', region, '-']
    result = subprocess.run(command, capture_output=True, check=True)
    data_start = result.stdout.find(b'\n', result.stdout.find(b'\n', result.stdout.find(b'\n') + 1) + 1) + 1
    flat_array = np.frombuffer(result.stdout[data_start:], dtype=np.uint8)
    img = flat_array.reshape((rel_height, rel_width, 3))
    
    return img

# Example: If your Waydroid is at 0,0 and you set it to 1280x720
width = 960
height = 540
screen_width = 1920
screen_height = 1080
taskbar_offset = 18
x1 = (screen_width - width) // 2 
y1 = (screen_height - height) // 2 + taskbar_offset


# img = capture_waydroid_optimized_np(x1, y1, width, height)
start = time.perf_counter()
img = take_screenshot()
print(time.perf_counter()-start)

if isinstance(img, np.ndarray):
    img = Image.fromarray(img)

nas_path = Path("/mnt/SF_NAS/Oliver")

img.save(nas_path / "image_problem4.jpg")