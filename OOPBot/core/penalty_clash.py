import time
import pyautogui
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from config.penalty_clash_config import (
    IMAGE_PATHS as PENALTY_IMAGE_PATHS,
    PENALTY_CLASH_ROI,
    UNSTOPPABLE_CROP,
    UNSTOPPABLE_LOW,
    UNSTOPPABLE_HIGH,
    GREEN_CROP,
    TRIANGLE_CROP,
    GREEN_CHANNEL_THRESHOLD,
    BLUE_CHANNEL_MAX,
    RED_CHANNEL_MAX,
    TRIANGLE_GRAY_THRESHOLD,
    ENERGY_ROI
)
import cv2
from utils.image_processing import find_and_click
from utils.logging_utils import BotLogger
import pytesseract
import win32api
import win32con
import mss

# TODO: check if pyautogui inbuilt functions work faster than the code you currently have
# TODO: try to figure out why the code is so much slower here than in the separate python scripts. Is it just because ldplayer is running?

class PenaltyClashBot:
    def __init__(self, team_name: str):
        self.team_name = team_name
        self.logger = BotLogger(__name__)

    def run(self):
        self.logger.info(f"Starting PenaltyClashBot for team: {self.team_name}")
        steps = [
            # (PENALTY_IMAGE_PATHS["home"], "home button"),
            (PENALTY_IMAGE_PATHS["events"], "events button"),
            (PENALTY_IMAGE_PATHS["go"], "go button"),
            (PENALTY_IMAGE_PATHS["play"], "play button"),
        ]
        for template, desc in steps:
            self.logger.info(f"Looking for {desc}...")
            if not find_and_click(str(template), description=desc):
                self.logger.error(f"Failed to find and click {desc}. Aborting.")
                return
            self.logger.info(f"Clicked {desc}.")
            time.sleep(2)
        self.logger.info("Moving mouse to (1700, 900)")
        pyautogui.moveTo(1700, 900, duration=0.5)
        self.logger.info("Starting unstoppable/triangle detection loop.")
        try:
            while True:
                self._taking_shot_phase()
                
                # Check if match has ended
                self.logger.info("Checking for 'done' template...")
                if find_and_click(str(PENALTY_IMAGE_PATHS["done"]), description="done button"):
                    self.logger.info("Match ended. Clicked 'done' button.")
                    time.sleep(2)
                    
                    # Check energy with OCR
                    self.logger.info("Checking remaining energy...")
                    screenshot = self._take_screenshot(ENERGY_ROI)
                    arr = np.array(screenshot)
                    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    energy_text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
                    
                    try:
                        energy_value = int(energy_text)
                        self.logger.info(f"Energy remaining: {energy_value}")
                        
                        if energy_value == 0:
                            self.logger.info("No energy remaining. Exiting bot.")
                            return
                        else:
                            self.logger.info(f"Energy remaining: {energy_value}. Starting new match.")
                            if find_and_click(str(PENALTY_IMAGE_PATHS["play"]), description="play button"):
                                self.logger.info("Clicked play button for new match.")
                                continue
                            else:
                                self.logger.error("Could not find play button for new match. Exiting.")
                                return
                    except ValueError:
                        self.logger.error(f"Could not parse energy value from text: '{energy_text}'. Exiting.")
                        return
                else:
                    self.logger.debug("No 'done' template found, continuing with current match.")
        except Exception as e:
            self.logger.error("Error in penalty clash loop", error=e)

    def _taking_shot_phase(self):
        self.logger.info("Waiting for unstoppable pixel detection...")
        count = 0
        while True:
            start_screenshot = time.perf_counter()
            screenshot = self._take_screenshot(UNSTOPPABLE_CROP)
            arr = np.array(screenshot)
            cropped = arr  # Already cropped by mss

            if self._find_matching_pixel_fast(cropped, UNSTOPPABLE_LOW, UNSTOPPABLE_HIGH):
                break
 
            count += 1

        self.logger.info(f"{time.perf_counter()-start_screenshot}s reaction time")
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.logger.info("Unstoppable pixel detected. Proceeding to click and hold.")
        try:
            self.logger.info("Waiting for triangle detection in white pixel range...")
            count = 0
            while True:
                start_screenshot = time.perf_counter()
                screenshot = self._take_screenshot(GREEN_CROP)
                arr = np.array(screenshot)
                green = arr  # Already cropped by mss
                
                # Vectorized green pixel detection - much faster than separate channel operations
                green_mask = (
                    (green[:, :, 1] > GREEN_CHANNEL_THRESHOLD) &
                    (green[:, :, 2] < BLUE_CHANNEL_MAX) &
                    (green[:, :, 0] < RED_CHANNEL_MAX)
                )
                
                # Early exit if no green pixels found
                if not np.any(green_mask):
                    continue
                    
                # Find white pixel range using vectorized operations
                white_pixels = np.column_stack(np.where(green_mask))
                leftmost_x = np.min(white_pixels[:, 1]) + GREEN_CROP["x1"]
                rightmost_x = np.max(white_pixels[:, 1]) + GREEN_CROP["x1"]
                
                triangle_screenshot = self._take_screenshot(TRIANGLE_CROP)
                triangle = np.array(triangle_screenshot)
                gray = cv2.cvtColor(triangle, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, TRIANGLE_GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                M = cv2.moments(approx)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"]) + TRIANGLE_CROP["x1"]
                self.logger.debug(f"Triangle centroid x: {cx}, white pixel range: {leftmost_x}-{rightmost_x}")
                if leftmost_x < cx < rightmost_x:
                    self.logger.info("Triangle detected in white pixel range. Releasing mouse.")
                    break
                elif leftmost_x < cx:
                    self.logger.info("Triangle detected past start of pixel range, but too late released")
                    break
                count += 1
        finally:
            self.logger.info(f"{time.perf_counter() - start_screenshot}s reaction time")
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.logger.info("Mouse released.")

    def _find_matching_pixel_fast(self, img: np.ndarray, target_rgb_low: Tuple[int, int, int], target_rgb_high: Tuple[int, int, int]):
        # Vectorized pixel search - much faster than nested loops
        low = np.array(target_rgb_low)
        high = np.array(target_rgb_high)
        
        # Sample every 5th pixel for speed
        sampled = img[::5, ::5]
        
        # Vectorized comparison
        mask = np.all((sampled >= low) & (sampled <= high), axis=2)
        return np.any(mask)


    def _take_screenshot(self, region=None) -> Image.Image:
        # Take a screenshot using mss and return as PIL Image. If region is provided, only capture that region.
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            if region is not None:
                # region dict: {"x1": left, "y1": top, "x2": right, "y2": bottom}
                mon = {
                    "left": region["x1"],
                    "top": region["y1"],
                    "width": region["x2"] - region["x1"],
                    "height": region["y2"] - region["y1"]
                }
                sct_img = sct.grab(mon)
            else:
                sct_img = sct.grab(monitor)
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        return img 