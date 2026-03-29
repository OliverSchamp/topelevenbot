"""
Ad watching bot functionality for Top Eleven
"""

import time
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
from enum import Enum
from typing import List, Optional

from utils.logging_utils import BotLogger
from utils.image_processing import find_and_click, crop_black_bars, fast_click
# from utils.image_processing import take_screenshot_fast as take_screenshot
from utils.image_processing import take_screenshot
from config.ad_config import (
    CLICK_DELAY,
    AD_CHECK_INTERVAL,
    IMAGE_PATHS,
    MAX_TIME_WITHOUT_X
)
from config.general_config import ocr_pipeline, mouse_keyboard_controller
from utils.ocr import OCRResult
from interface import TemplateMatch, ScreenRegion
from utils.x_button_ai import detect_x_buttons

training_data_dir = Path("img/x_examples/training_data")

image_id = 0
for image in training_data_dir.glob("*.jpg"):
    image_id = max(image_id, int(image.stem))

class AdWatchState(Enum):
    UNDEFINED = 0
    INIT = 1
    SHOP = 2
    WATCHING = 3
    GSTORE = 4
    RESTART = 5
    EXIT = 6

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

        # new adwatch
        self.state = AdWatchState.INIT
        self.time_started_watching = 0.0
        self.dynamic_detection_occurred = False
        self.click_ads_tries = 0
    
    def _navigate_to_green_hud(self) -> bool:
        """Navigate to the green HUD screen"""
        try:
            # Find and click green HUD
            if not find_and_click(str(IMAGE_PATHS['green_hud']), description="green HUD"):
                self.logger.error("Could not find green HUD")
                self.should_restart = True
                return False
            
            time.sleep(1)  # Wait for screen to load
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
            
            fast_click(center_x, center_y)
            time.sleep(CLICK_DELAY)
            
            return True
            
        except Exception as e:
            self.logger.error("Error clicking X button", e)
            return False
    
    def _compare_screenshots(self, screenshot, prev_screenshot, epsilon=5) -> bool:
        if prev_screenshot is not None and screenshot.shape == prev_screenshot.shape:
            diff = np.mean(np.abs(screenshot.astype(np.float32) - prev_screenshot.astype(np.float32)))
            return diff < epsilon

    def dynamic_x_detection_2(self, screenshot) -> Optional[ScreenRegion]:
        """
        If the screenshot is similar to the previous and no X is detected, rerun detection with low threshold and click the highest-confidence detection if any.
        Returns True if a click was made, False otherwise.
        """

        screenshot_cropped, (offset_x, offset_y) = crop_black_bars(screenshot)
        if offset_x < 10 and offset_y < 10:
            all_regions = detect_x_buttons(screenshot, conf_threshold=0.001)

            if len(all_regions) == 0:
                return None

            region = max(all_regions, key = lambda r: r.conf)
        else:
            all_regions = detect_x_buttons(screenshot_cropped, conf_threshold=0.001)

            if len(all_regions) == 0:
                return None

            region = max(all_regions, key = lambda r: r.conf)
            region = ScreenRegion(
                x1=region.x1 + offset_x,
                x2=region.x2 + offset_x,
                y1=region.y1 + offset_y,
                y2=region.y2 + offset_y
            )

        return region

    def set_state(self, new_state):
        self.logger.info(f"{self.state} -> {new_state}")
        self.state = new_state
    
    def read_screen(self) -> OCRResult:
        return ocr_pipeline.run(take_screenshot())

    def limit_reached(self, ocr_result: OCRResult) -> bool:
        return ocr_result.contains_all(["limit", "reached"])

    def detect_shop(self, ocr_result: OCRResult) -> bool:
        return ocr_result.contains_all(["shop", "offers", "currencies", "club", "items"])

    def detect_gstore(self, ocr_result: OCRResult) -> bool:
        return ocr_result.contains_all(["games", "apps", "search", "books", "you", "kids"])
    
    def click_ads_button(self) -> bool:
        return find_and_click(IMAGE_PATHS["greens_ads_button"], description="greens ads button")
    
    def time_spent_watching(self) -> float:
        return time.time() - self.time_started_watching

    def click_max_x_region(self, x_regions: List[ScreenRegion]):
        max_region = max(x_regions, key = lambda r: r.conf)
        center_x = (max_region.x1 + max_region.x2) // 2
        center_y = (max_region.y1 + max_region.y2) // 2
        fast_click(center_x, center_y)
    
    def look_for_x_button(self) -> List[ScreenRegion]:
        # If not in shop, watch the ad
        screenshot = take_screenshot()
        x_regions = detect_x_buttons(screenshot)

        if len(x_regions) == 0:
            screenshot_cropped, (offset_x, offset_y) = crop_black_bars(screenshot, black_thresh=30)
            if offset_x != 0 or offset_y != 0:
                try:
                    x_regions = detect_x_buttons(screenshot_cropped)
                except ZeroDivisionError:
                    return []
                # Translate all region coordinates back to original image
                x_regions = [ScreenRegion(
                    x1=r.x1 + offset_x,
                    x2=r.x2 + offset_x,
                    y1=r.y1 + offset_y,
                    y2=r.y2 + offset_y,
                    conf=r.conf
                ) for r in x_regions]
        
        return x_regions
    
    def _watch_ad_loop(self) -> AdWatchState:
        # TODO: I think there's something in the state machine that stops it from 
        init_count = 0
        while self.state == AdWatchState.INIT:
            if self.detect_shop(self.read_screen()):
                self.set_state(AdWatchState.SHOP)
            init_count += 1

            if init_count > 30:
                self.set_state(AdWatchState.RESTART)
                self.logger.info("Could not start the ad watch loop")
        
        while True:
            self.logger.info(f"Starting new iteration...")
            ocr_result = self.read_screen()

            if self.detect_shop(ocr_result):
                self.logger.info("Detected shop")
                self.set_state(AdWatchState.SHOP)
            elif self.detect_gstore(ocr_result):
                self.set_state(AdWatchState.GSTORE)

            if self.state == AdWatchState.SHOP:
                if self.limit_reached(ocr_result):
                    self.set_state(AdWatchState.EXIT)
                    break

                if self.click_ads_button():
                    self.set_state(AdWatchState.WATCHING)
                    self.time_started_watching = time.time()
                    self.dynamic_detection_occurred = False # reset all the parameters once back in shop before beginning new cycle
                    self.click_ads_tries = 0
                else:
                    time.sleep(1)
                    self.click_ads_tries += 1

                    if self.click_ads_tries > 60:
                        self.set_state(AdWatchState.RESTART) # button not found, restart game
                        break
                    self.logger.info("Inside shop, waiting for ad button to load...")
                    continue
            ##########################################################
            while self.state == AdWatchState.GSTORE:
                if self.detect_gstore(ocr_result):
                    mouse_keyboard_controller.press_key("esc")
                    time.sleep(2)
                else:
                    time.sleep(2)
                    if self.detect_shop(ocr_result):
                        self.set_state(AdWatchState.SHOP)
                    else:
                        self.set_state(AdWatchState.WATCHING)
                
                ocr_result = self.read_screen()
            
            if self.state == AdWatchState.SHOP:
                continue
            
            ##########################################################
            
            assert self.state == AdWatchState.WATCHING

            if self.time_spent_watching() > 120:
                if not self.dynamic_detection_occurred:
                    x_regions = self.dynamic_x_detection_2(take_screenshot())
                    if x_regions is not None:
                        self.click_max_x_region(x_regions)
                    self.dynamic_detection_occurred = True
                    time.sleep(2)
                else:
                    self.set_state(AdWatchState.RESTART)
                    break # still in the ad after 2 minutes, dynamic detection hasnt worked, should restart game
            elif self.time_spent_watching() > 20:
                mouse_keyboard_controller.press_key("esc")
                time.sleep(2)

            ##########################################################

            x_regions = self.look_for_x_button()
            count = 0
            while len(x_regions): # TODO: check if X is also in same location
                time.sleep(2)
                count += 1
                self.logger.info(f"X detected. Count: {count}")
                if count >= 2:
                    if len(x_regions) > 1:
                        confs = [r.conf for r in x_regions]
                        confs.sort(reverse=True)
                        if confs[0] - confs[1] < 0.05:  
                            best_lw_ratio = 0
                            squarest_region = x_regions[0]
                            for x_region in x_regions:
                                if np.abs(x_region.wh_ratio() - 1) < np.abs(best_lw_ratio - 1):
                                    best_lw_ratio = x_region.wh_ratio()
                                    squarest_region = x_region

                            # self.click_max_x_region(squarest_region)
                            center_x = (squarest_region.x1 + squarest_region.x2) // 2
                            center_y = (squarest_region.y1 + squarest_region.y2) // 2
                            fast_click(center_x, center_y)
                            break
                        else:
                            self.click_max_x_region(x_regions)
                            break
                    else:
                        self.click_max_x_region(x_regions)
                        break
                
                x_regions = self.look_for_x_button()

            time.sleep(3)

        return self.state  #dont really need to do this... but just to remind you

    def _save_screenshot(self, screenshot: np.ndarray, filename: str) -> None:
        """Save screenshot to file"""
        cv2.imwrite(f"/mnt/SF_NAS/Oliver/Log/{filename}", cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
    
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
                
                time.sleep(1.5) # wait for the app to refresh

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