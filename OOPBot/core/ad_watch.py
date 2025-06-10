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
from utils.image_processing import find_and_click, find_on_screen, take_screenshot
from config.ad_config import (
    CLICK_DELAY,
    AD_CHECK_INTERVAL,
    CONFIDENCE_THRESHOLD,
    IMAGE_PATHS,
    X_BUTTONS_DIR,
    MAX_TIME_WITHOUT_X
)
from interface import TemplateMatch, ScreenRegion

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
    
    def _check_for_x_button(self) -> Optional[ScreenRegion]:
        """Check for any X button templates in the top right quarter of the screen"""
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        
        # Define top right quarter region
        search_region = ScreenRegion(
            x1=screen_width//2,
            y1=0,
            x2=screen_width,
            y2=screen_height//2
        )
        
        # Check each X button template
        x_button_path = Path(X_BUTTONS_DIR)
        for template in x_button_path.glob("*.jpg"):
            match = find_on_screen(
                str(template),
                threshold=CONFIDENCE_THRESHOLD-0.1,
                description=f"X button ({template.name})",
                search_region=search_region
            )
            
            if match.center_x is not None and match.center_y is not None:
                self.logger.info(f"Found X button at ({match.center_x}, {match.center_y})")
                self.last_x_time = time.time()
                return ScreenRegion(
                    x1=match.center_x - 5,
                    y1=match.center_y - 5,
                    x2=match.center_x + 5,
                    y2=match.center_y + 5
                )
        
        return None
    
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
    
    def _watch_ad_loop(self) -> bool:
        """Execute the main ad watching loop"""
        try:

            while True:
                find_and_click(str(IMAGE_PATHS['greens_ads_button']), description="greens ads button")

                # Check for X button
                x_region = self._check_for_x_button()
                
                if x_region is not None:
                    self.logger.info("Found X button, attempting to click it")
                    if not self._click_x_button(x_region):
                        self.logger.error("Failed to click X button")
                        self.should_restart = True
                        return False
                    
                    # Wait before next check
                    time.sleep(1)
                    continue
                
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