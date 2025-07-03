"""
Ad watching bot functionality for Top Eleven
"""

import time
from pathlib import Path
from typing import Optional, Tuple, List
import pyautogui
import cv2
import numpy as np

from utils.logging_utils import BotLogger
from utils.image_processing import find_and_click, find_on_screen, take_screenshot, crop_black_bars
from config.ad_config import (
    CLICK_DELAY,
    AD_CHECK_INTERVAL,
    CONFIDENCE_THRESHOLD,
    IMAGE_PATHS,
    X_BUTTONS_DIR,
    MAX_TIME_WITHOUT_X
)
from interface import TemplateMatch, ScreenRegion
from utils.x_button_ai import detect_x_buttons

training_data_dir = Path("img/x_examples/training_data")

image_id = 0
for image in training_data_dir.glob("*.jpg"):
    image_id = max(image_id, int(image.stem))


class AdWatchResult:
    """Class to represent ad watching result"""
    SUCCESS = 'success'
    ERROR = 'error'
    RESTART_NEEDED = 'restart_needed'

class AdWatchBot:
    """Bot for watching ads"""
    
    def __init__(self, team_name: str):
        """Initialize ad watch bot"""
        self.team_name = team_name
        self.logger = BotLogger(__name__)
        self.should_restart = False
        self.last_x_time = time.time()
        
        # Verify X button templates exist
        x_button_path = Path(X_BUTTONS_DIR)
        if not x_button_path.exists() or not any(x_button_path.glob("*.jpg")):
            self.logger.error("No X button templates found in img/x_examples/")
            raise FileNotFoundError("No X button templates found")
    
    def _navigate_to_green_hud(self) -> bool:
        """Navigate to the green HUD screen"""
        try:
            # Find and click green HUD
            if not find_and_click(str(IMAGE_PATHS['green_hud']), description="green HUD"):
                self.logger.error("Could not find green HUD")
                self.should_restart = True
                return False
            
            time.sleep(3)  # Wait for screen to load
            return True
            
        except Exception as e:
            self.logger.error("Error navigating to green HUD", e)
            self.should_restart = True
            return False
    

    
    def _click_x_button(self, x_region: ScreenRegion) -> bool:
        """Click the X button in the specified region"""
        try:
            # Move to center of region and click
            center_x = (x_region.x1 + x_region.x2) // 2
            center_y = (x_region.y1 + x_region.y2) // 2
            
            pyautogui.moveTo(center_x, center_y, duration=0.5)
            pyautogui.click()
            time.sleep(CLICK_DELAY)
            
            return True
            
        except Exception as e:
            self.logger.error("Error clicking X button", e)
            return False
    
    def _dynamic_x_detection(self, screenshot, prev_screenshot, epsilon=5):
        """
        If the screenshot is similar to the previous and no X is detected, rerun detection with low threshold and click the highest-confidence detection if any.
        Returns True if a click was made, False otherwise.
        """
        if prev_screenshot is not None and screenshot.shape == prev_screenshot.shape:
            diff = np.mean(np.abs(screenshot.astype(np.float32) - prev_screenshot.astype(np.float32)))
            if diff < epsilon:
                # 1. Run model on original screenshot
                all_regions = detect_x_buttons(screenshot, conf_threshold=0.05)
                if len(all_regions) == 1:
                    region = all_regions[0]
                    self.logger.info(f"[Dynamic] Clicking X with highest confidence at ({(region.x1+region.x2)//2}, {(region.y1+region.y2)//2}) [original image]")
                    center_x = (region.x1 + region.x2) // 2
                    center_y = (region.y1 + region.y2) // 2
                    pyautogui.moveTo(center_x, center_y, duration=0.5)
                    pyautogui.click()
                    time.sleep(1)
                    return True
                # 2. If not, try with cropped image
                screenshot_cropped, (offset_x, offset_y) = crop_black_bars(screenshot)
                all_regions = detect_x_buttons(screenshot_cropped, conf_threshold=0.05)
                if len(all_regions) == 1:
                    region = all_regions[0]
                    # Translate region coordinates back to original image
                    region = ScreenRegion(
                        x1=region.x1 + offset_x,
                        x2=region.x2 + offset_x,
                        y1=region.y1 + offset_y,
                        y2=region.y2 + offset_y
                    )
                    self.logger.info(f"[Dynamic] Clicking X with highest confidence at ({(region.x1+region.x2)//2}, {(region.y1+region.y2)//2}) [cropped image]")
                    center_x = (region.x1 + region.x2) // 2
                    center_y = (region.y1 + region.y2) // 2
                    pyautogui.moveTo(center_x, center_y, duration=0.5)
                    pyautogui.click()
                    time.sleep(1)
                    return True
        return False

    def _watch_ad_loop(self) -> bool:
        """Execute the main ad watching loop"""
        try:
            saw_object_last_time = False
            prev_screenshot = None
            epsilon = 5  # Similarity threshold for screenshots
            while True:
                time.sleep(1)  # 1. Wait 1 second before each loop

                #find and click on the greens_ads_button template
                find_and_click(IMAGE_PATHS["greens_ads_button"])

                # 2. Template match for ldplayer_suggest - which is the suggestion to download the ad game in the ld store
                match = find_on_screen(
                    IMAGE_PATHS['ldplayer_suggest'],
                    threshold=0.8,  # You may adjust this threshold
                    description="ldplayer_suggest"
                )
                if match.top_left_x is not None and match.top_left_y is not None:
                    # Click at (top_left_x + 320, top_left_y + 210)
                    click_x = match.top_left_x + 320
                    click_y = match.top_left_y + 210
                    self.logger.info(f"Clicking ldplayer_suggest at ({click_x}, {click_y})")
                    pyautogui.moveTo(click_x, click_y, duration=0.5)
                    pyautogui.click()
                    time.sleep(1)
                    continue  # After clicking, continue to next loop iteration

                # 2.5 Template match for exit ad suggest
                match = find_on_screen(
                    IMAGE_PATHS['ads_exit_suggest'],
                    threshold=0.8,  # You may adjust this threshold
                    description="ads_exit_suggest"
                )
                if match.top_left_x is not None and match.top_left_y is not None:
                    # Click at (top_left_x + 320, top_left_y + 210)
                    click_x = match.top_left_x + 500
                    click_y = match.top_left_y + 360
                    self.logger.info(f"Clicking ads_exit_suggest at ({click_x}, {click_y})")
                    pyautogui.moveTo(click_x, click_y, duration=0.5)
                    pyautogui.click()
                    time.sleep(1)
                    continue  # After clicking, continue to next loop iteration

                # Take screenshot for X detection
                screenshot = take_screenshot()
                # 1. Run model on original screenshot
                x_regions = detect_x_buttons(screenshot)
                if len(x_regions) != 1:
                    # 2. If not, try with cropped image
                    screenshot_cropped, (offset_x, offset_y) = crop_black_bars(screenshot)
                    try:
                        x_regions = detect_x_buttons(screenshot_cropped)
                    except ZeroDivisionError:
                        continue
                    # Translate all region coordinates back to original image
                    x_regions = [ScreenRegion(
                        x1=r.x1 + offset_x,
                        x2=r.x2 + offset_x,
                        y1=r.y1 + offset_y,
                        y2=r.y2 + offset_y
                    ) for r in x_regions]

                # 4. If more than one detection, treat as no detection
                if len(x_regions) != 1:
                    # Dynamic confidence thresholding if screenshot is similar to previous
                    time_since_last_x = time.time() - self.last_x_time
                    if self._dynamic_x_detection(screenshot, prev_screenshot, epsilon) and time_since_last_x > 60:
                        saw_object_last_time = False
                        prev_screenshot = screenshot
                        self.last_x_time = time.time()
                        continue
                    saw_object_last_time = False
                else:
                    # 3. Only click if saw object last time too
                    if saw_object_last_time:
                        self.logger.info("Clicking detected X button (AI model)")
                        center_x = (x_regions[0].x1 + x_regions[0].x2) // 2
                        center_y = (x_regions[0].y1 + x_regions[0].y2) // 2
                        pyautogui.moveTo(center_x, center_y, duration=0.5)
                        pyautogui.click()
                        self.last_x_time = time.time()
                        continue
                    saw_object_last_time = True

                prev_screenshot = screenshot

                # Check if we've gone too long without seeing an X button
                time_since_last_x = time.time() - self.last_x_time
                if time_since_last_x > MAX_TIME_WITHOUT_X:
                    self.logger.warning(f"No X button found for {time_since_last_x:.1f} seconds, restarting")
                    self.should_restart = True
                    return False

                # Wait before next check
                time.sleep(AD_CHECK_INTERVAL)
        except Exception as e:
            self.logger.error("Error in ad watching loop", e)
            self.should_restart = True
            return False
    
    def run(self) -> None:
        """Main ad watching loop"""
        try:
            while True:
                self.should_restart = False
                
                # Navigate to green HUD
                if not self._navigate_to_green_hud():
                    if self.should_restart:
                        self._prepare_restart()
                        return
                    continue
                
                # Start watching ads
                if not self._watch_ad_loop():
                    if self.should_restart:
                        self._prepare_restart()
                        return
                    continue
                
        except Exception as e:
            self.logger.error("Error in ad watching loop", e)
            self._prepare_restart()
    
    def _prepare_restart(self) -> None:
        """Prepare for bot restart"""
        self.logger.info("Preparing for restart")
        # Exit fullscreen
        pyautogui.press('f11')
        time.sleep(0.5) 