"""
Utility functions for image processing and screen interaction
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Union, Dict, List
import logging
import subprocess

from config.auction_config import CONFIDENCE_THRESHOLD
from config.general_config import SCREEN_HEIGHT, SCREEN_WIDTH, width, height, taskbar_offset_px, mouse_keyboard_controller, mouse_move_controller, screengrabber
from interface import TemplateMatch, ScreenRegion

logger = logging.getLogger(__name__)


def get_fixed_boundaries(points_dict: Dict[str, float], start_bound: int) -> Dict[str, Tuple[int, int]]:
    # sort the points by their x-coordinate
    points_dict = dict(sorted(points_dict.items(), key=lambda item: item[1]))
    points = list(points_dict.values())
    initial_radius = points[0] - start_bound
    output = {list(points_dict.keys())[0]: (start_bound, points[0] + initial_radius)}
    for k, v in points_dict.items():
        if k == list(points_dict.keys())[0]:
            continue
        prev_point = output[list(output.keys())[-1]][-1]
        current_point = v
        radius = (current_point - prev_point)
        output[k] = (current_point - radius, current_point + radius)
    return output


def crop_black_bars(image, black_thresh=30, min_bar_thickness_ratio=0.05):
    """
    Detect and crop large black bars (letterbox) from the image.
    Args:
        image: Input BGR image (numpy array)
        black_thresh: Pixel value threshold to consider as black (0-255)
        min_bar_thickness_ratio: Minimum thickness ratio (w.r.t. image size) to consider as a bar
    Returns:
        cropped_image: Cropped image (if bars found), else original image
        offsets: (left, top) pixel offsets of the crop
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    h, w = mask.shape
    col_sum = np.sum(mask, axis=0) / 255
    row_sum = np.sum(mask, axis=1) / 255
    col_black_thresh = h * 0.98
    row_black_thresh = w * 0.98
    left = 0
    while left < w and col_sum[left] > col_black_thresh:
        left += 1
    right = w - 1
    while right >= 0 and col_sum[right] > col_black_thresh:
        right -= 1
    top = 0
    while top < h and row_sum[top] > row_black_thresh:
        top += 1
    bottom = h - 1
    while bottom >= 0 and row_sum[bottom] > row_black_thresh:
        bottom -= 1
    min_bar_w = int(w * min_bar_thickness_ratio)
    min_bar_h = int(h * min_bar_thickness_ratio)
    crop_left = left if left > min_bar_w else 0
    crop_right = right if (w - 1 - right) > min_bar_w else w - 1
    crop_top = top if top > min_bar_h else 0
    crop_bottom = bottom if (h - 1 - bottom) > min_bar_h else h - 1
    if crop_left > 0 or crop_right < w - 1 or crop_top > 0 or crop_bottom < h - 1:
        cropped = image[crop_top:crop_bottom+1, crop_left:crop_right+1]
        return cropped, (crop_left, crop_top)
    return image, (0, 0)

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

def take_screenshot_fast(x1: int = 0, y1: int = 0, x2: int = width, y2: int = height, mode: str = "RGB"):
    img = screengrabber.grab_frame(mode)
    return img[y1:y2, x1:x2]

def find_on_screen(
    template_path: str, 
    threshold: float = CONFIDENCE_THRESHOLD, 
    description: str = "element",
    search_region: Optional[ScreenRegion] = None
) -> TemplateMatch:
    """
    Look for an image on the screen
    Args:
        template_path: Path to the template image
        threshold: Confidence threshold for matching
        description: Description of what we're looking for (for logging)
        search_region: Optional region to search in
    Returns: TemplateMatch object containing match details and confidence score
    """
    try:
        template = cv2.cvtColor(cv2.imread(str(template_path)), cv2.COLOR_BGR2RGB)
        if template is None:
            raise FileNotFoundError(f"Could not load template image: {template_path}")
        
        screenshot = take_screenshot()
        
        # If search region is specified, crop the screenshot
        if search_region:
            screenshot = screenshot[
                search_region.y1:search_region.y2,
                search_region.x1:search_region.x2
            ]

        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            w, h = template.shape[1], template.shape[0]
            top_left_x, top_left_y = max_loc
            
            # Adjust coordinates if using search region
            if search_region:
                top_left_x += search_region.x1
                top_left_y += search_region.y1
            
            center_x = top_left_x + w//2
            center_y = top_left_y + h//2
            logger.info(f"Found {description} with confidence: {max_val:.2%}")
            
            return TemplateMatch(
                center_x=center_x,
                center_y=center_y,
                top_left_x=top_left_x,
                top_left_y=top_left_y,
                width=w,
                height=h,
                confidence=max_val
            )
        else:
            logger.info(f"No {description} found as confidence score is {max_val:.2%}")
            cv2.imwrite(f"img/auto_auction/no_{description}.jpg", screenshot)
            cv2.imwrite(f"img/auto_auction/{description}_template.jpg", template)


            return TemplateMatch(
                center_x=None,
                center_y=None,
                top_left_x=None,
                top_left_y=None,
                width=None,
                height=None,
                confidence=max_val
            )
            
    except Exception as e:
        logger.error(f"Error finding {description} on screen: {str(e)}")
        return TemplateMatch(
            center_x=None,
            center_y=None,
            top_left_x=None,
            top_left_y=None,
            width=None,
            height=None,
            confidence=0.0
        )

def fast_click(x, y):
    x_dest = ((SCREEN_WIDTH - width) // 2) + x
    y_dest = ((SCREEN_HEIGHT - height) // 2) + y + taskbar_offset_px
    mouse_move_controller.click(x_dest, y_dest, "nothing")
    mouse_keyboard_controller.left_click()

def fast_move(x, y):
    x_dest = ((SCREEN_WIDTH - width) // 2) + x
    y_dest = ((SCREEN_HEIGHT - height) // 2) + y + taskbar_offset_px
    mouse_move_controller.click(x_dest, y_dest, "nothing")

def press_only():
    mouse_keyboard_controller.left_mouse_down()

def release_only():
    mouse_keyboard_controller.left_mouse_up()

def find_and_click(
    template_path: str, 
    threshold: float = CONFIDENCE_THRESHOLD, 
    description: str = "button",
    click_delay: float = 0.2,
    search_region: Optional[ScreenRegion] = None
) -> bool:
    """
    Look for an image on the screen and click it if found
    Args:
        template_path: Path to the template image
        threshold: Confidence threshold for matching
        description: Description of what we're looking for (for logging)
        click_delay: Delay after clicking in seconds
        search_region: Optional region to search in
    Returns: True if found and clicked, False otherwise
    """
    logger.debug(f"Searching for {description}...")
    
    match = find_on_screen(template_path, threshold, description, search_region)
    logger.debug(f"Confidence score for {description}: {match.confidence:.2%}")
    
    if match.center_x is not None and match.center_y is not None:
        logger.debug(f"Moving mouse to {description} at coordinates: ({match.center_x}, {match.center_y})")
        fast_click(match.center_x, match.center_y)
        time.sleep(click_delay)
        return True
    
    return False

def safe_int_convert(value: Union[str, int]) -> Optional[int]:
    """
    Safely converts a string to integer, handling leading zeros.
    Returns None if conversion fails.
    """
    try:
        if isinstance(value, str):
            value = value.strip()
            value = value.lstrip('0')
            if not value:
                return 0
        return int(value)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting {value} to integer: {str(e)}")
        return None 