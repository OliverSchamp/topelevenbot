"""
Utility functions for image processing and screen interaction
"""

import cv2
import numpy as np
import pyautogui
import time
from typing import Optional, Tuple, Union
import logging

from config.auction_config import CONFIDENCE_THRESHOLD
from interface import TemplateMatch, ScreenRegion

logger = logging.getLogger(__name__)

def crop_black_bars(image, black_thresh=10, min_bar_thickness_ratio=0.05):
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

def take_screenshot() -> np.ndarray:
    """Take a screenshot of the entire screen"""
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

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
        template = cv2.imread(str(template_path))
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

def find_and_click(
    template_path: str, 
    threshold: float = CONFIDENCE_THRESHOLD, 
    description: str = "button",
    click_delay: float = 0.5,
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
    logger.info(f"Searching for {description}...")
    
    match = find_on_screen(template_path, threshold, description, search_region)
    logger.info(f"Confidence score for {description}: {match.confidence:.2%}")
    
    if match.center_x is not None and match.center_y is not None:
        logger.info(f"Moving mouse to {description} at coordinates: ({match.center_x}, {match.center_y})")
        pyautogui.moveTo(match.center_x, match.center_y, duration=0.5)
        pyautogui.click()
        time.sleep(click_delay)
        return True
    
    return False

def clock_image_preprocessing(image: np.ndarray) -> np.ndarray:
    """Preprocess clock image for better OCR results"""
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    processed_image = cv2.inRange(processed_image, (35, 0, 0), (77, 255, 255))
    kernel = np.ones((3,3), np.uint8)
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    return processed_image

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