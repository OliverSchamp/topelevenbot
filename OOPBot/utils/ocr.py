"""
OCR utility functions for text extraction from images
"""

import cv2
import pytesseract
import re
import logging
import numpy as np
from typing import Optional

from config.auction_config import TESSERACT_CMD

# Set the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logger = logging.getLogger(__name__)

def extract_text_from_region(image: np.ndarray, preprocess: bool = True) -> str:
    """Extract text from image region using OCR"""
    try:
        if preprocess:
            # Convert to grayscale if image has 3 channels
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply thresholding
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            # Check the dimensions of the image. if the image is 3 dimensional with 3 channels, convert to grayscale
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
        text = pytesseract.image_to_string(gray, config='--psm 6')
        return text.strip()
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {str(e)}")
        return ""

def extract_numeric_value(text: str, money: bool = False) -> Optional[float]:
    """Extract numeric value and handle K/M conversion"""
    try:
        # Remove all non-numeric characters except K, M, and decimal point
        cleaned = ''.join(c for c in text if c.isdigit() or c in 'KM.')
        
        # Find number and unit
        match = re.match(r'(\d+\.?\d*)([KM])?', cleaned)
        if not match:
            return None
            
        number = float(match.group(1))
        unit = match.group(2)
        
        if money:
            # Convert K to M if necessary
            if unit == 'K':
                return number / 1000
            if unit == None:
                return number / 1000000
        return number
    except Exception as e:
        logger.error(f"Error extracting numeric value from text '{text}': {str(e)}")
        return None

def get_player_name(name_text: str) -> Optional[str]:
    """Extract player name from the specified region"""
    try:
        logger.info(f"Found name: {name_text}")
        last_dot_index = name_text.rfind('.')
        last_comma_index = name_text.rfind(',')
        
        # Find the rightmost dot/comma
        if last_dot_index > last_comma_index:
            split_index = last_dot_index
        else:
            split_index = last_comma_index
            
        # If no dot/comma found, or it is the first character, return original
        if split_index <= 0:
            return name_text.strip()
            
        # Get one char before and everything after the dot/comma
        output = name_text[split_index-1:].strip()
        logger.info(f"Extracted name: {output}")
        return output
        
    except Exception as e:
        logger.error(f"Error processing player name: {str(e)}")
        return None