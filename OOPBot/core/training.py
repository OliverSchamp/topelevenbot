"""
Training bot functionality for Top Eleven
"""

import time
import pyautogui
import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path
from typing import Optional, Tuple, Callable

from utils.logging_utils import BotLogger
from utils.image_processing import find_and_click, find_on_screen, take_screenshot
from config.training_config import (
    CLICK_DELAY,
    DRAG_DURATION,
    SCROLL_AMOUNT,
    MAX_SCROLL_ATTEMPTS,
    CONFIDENCE_THRESHOLD,
    PROMO_CONFIDENCE_THRESHOLD,
    PROGRESS_ROI,
    GREENS_BUDGET_ROI,
    MIN_CONDITION_THRESHOLD,
    MAX_RECOVERY_ATTEMPTS,
    RECOVERY_DELAY,
    TESSERACT_CMD,
    IMAGE_PATHS
)
from interface import TemplateMatch, ScreenRegion, TrainingProgress

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

class TrainingResult:
    """Class to represent training result"""
    SUCCESS = 'success'
    ERROR = 'error'
    RESTART_NEEDED = 'restart_needed'

class TrainingBot:
    """Bot for handling player training"""
    
    def __init__(self, team_name: str):
        """Initialize training bot"""
        self.team_name = team_name
        self.logger = BotLogger(__name__)
        self.should_restart = False
    
    def run(self) -> None:
        """Main training bot loop"""
        try:
            while True:
                self.should_restart = False
                
                # Navigate to training section
                if not self._navigate_to_training():
                    if self.should_restart:
                        self._prepare_restart()
                        return
                    continue
                
                # Select player and setup training
                if not self._setup_training():
                    if self.should_restart:
                        self._prepare_restart()
                        return
                    continue
                
                # Start training loop
                if not self._training_loop():
                    if self.should_restart:
                        self._prepare_restart()
                        return
                    continue
                
        except Exception as e:
            self.logger.error("Error in training loop", e)
            self._prepare_restart()
    
    def _navigate_to_training(self) -> bool:
        """Navigate to the training section"""
        try:
            # Click training section
            if not find_and_click(str(IMAGE_PATHS['training']), description="training section"):
                self.logger.error("Could not find training section")
                self.should_restart = True
                return False
            
            time.sleep(1)
            return True
            
        except Exception as e:
            self.logger.error("Error navigating to training section", e)
            self.should_restart = True
            return False
    
    def _find_player_and_confirm(self) -> bool:
        """Find player in list and confirm selection"""
        self.logger.info("Searching for player in list...")
        
        for attempt in range(MAX_SCROLL_ATTEMPTS):
            # Look for the player
            match = find_on_screen(
                str(IMAGE_PATHS['player_to_train']), 
                CONFIDENCE_THRESHOLD+0.1, 
                "player"
            )
            self.logger.info(f"Player search attempt {attempt + 1}, confidence: {match.confidence:.2%}")
            
            if match.center_x is not None:
                # Found the player, now look for confirm button
                confirm_match = find_on_screen(
                    str(IMAGE_PATHS['confirm_player']), 
                    CONFIDENCE_THRESHOLD, 
                    "confirm checkbox"
                )
                
                if confirm_match.center_x:
                    self.logger.info(f"Found confirm button at ({confirm_match.center_x}, {match.center_y})")
                    pyautogui.moveTo(confirm_match.center_x, match.center_y, duration=0.5)
                    pyautogui.click()
                    time.sleep(CLICK_DELAY)
                    return True
                    
            # Move mouse to center and drag down
            screen_width, screen_height = pyautogui.size()
            center_x = screen_width // 2
            center_y = screen_height // 2
            
            pyautogui.moveTo(center_x, center_y, duration=0.01)
            pyautogui.mouseDown()
            time.sleep(0.1)
            pyautogui.moveTo(center_x, center_y - 100, duration=0.2)  # Drag upwards 100 pixels
            time.sleep(0.1)
            pyautogui.mouseUp()
            time.sleep(0.5)
        
        return False
    
    def _drag_drill_to_slot(self) -> bool:
        """Drag a drill to an empty slot"""
        self.logger.info(f"Dragging drill to slot with threshold: {CONFIDENCE_THRESHOLD}")
        
        drill_template_match = find_on_screen(
            str(IMAGE_PATHS['warmup_drill']), 
            CONFIDENCE_THRESHOLD-0.2, 
            "drill"
        )
        drill_x = drill_template_match.center_x
        drill_y = drill_template_match.center_y


        slot_template_match = find_on_screen(
            str(IMAGE_PATHS['empty_slot']), 
            CONFIDENCE_THRESHOLD-0.2, 
            "empty slot"
        )
        slot_x = slot_template_match.center_x
        slot_y = slot_template_match.center_y
        
        if drill_x is not None and slot_x is not None:
            self.logger.info(f"Dragging drill from ({drill_x}, {drill_y}) to ({slot_x}, {slot_y})")
            pyautogui.moveTo(drill_x, drill_y, duration=0.5)
            pyautogui.mouseDown()
            time.sleep(0.1)
            pyautogui.moveTo(drill_x, drill_y+(slot_y-drill_y)/2, duration=DRAG_DURATION//10)
            pyautogui.moveTo(slot_x, slot_y, duration=DRAG_DURATION//10)
            time.sleep(0.1)
            pyautogui.mouseUp()
            time.sleep(CLICK_DELAY//10)
            return True
        
        return False
    
    def _setup_training(self) -> bool:
        """Setup training session with selected player and drills"""
        try:
            # Click players button
            if not find_and_click(str(IMAGE_PATHS['players']), description="players button"):
                self.logger.error("Could not find players button")
                self.should_restart = True
                return False
            
            # Find player and confirm
            if not self._find_player_and_confirm():
                self.logger.error("Could not find player or confirm button")
                self.should_restart = True
                return False
            
            # Click confirm
            if not find_and_click(str(IMAGE_PATHS['confirm']), description="confirm button"):
                self.logger.error("Could not find confirm button")
                self.should_restart = True
                return False
            
            # Click drills
            if not find_and_click(str(IMAGE_PATHS['drills']), description="drills button"):
                self.logger.error("Could not find drills button")
                self.should_restart = True
                return False
            
            # Click physical and mental
            if not find_and_click(str(IMAGE_PATHS['physical_n_mental']), description="physical and mental section"):
                self.logger.error("Could not find physical and mental section")
                self.should_restart = True
                return False
            
            # Fill empty slots with warmup drills
            while True:
                empty_slot_template_match = find_on_screen(
                    str(IMAGE_PATHS['empty_slot']), 
                    CONFIDENCE_THRESHOLD-0.2, 
                    "empty slot"
                )
                if empty_slot_template_match.confidence < CONFIDENCE_THRESHOLD-0.2:
                    break
                
                if not self._drag_drill_to_slot():
                    self.logger.error("Failed to drag drill to empty slot")
                    self.should_restart = True
                    return False
            
            # Click confirm
            if not find_and_click(str(IMAGE_PATHS['confirm']), description="final confirm button"):
                self.logger.error("Could not find final confirm button")
                self.should_restart = True
                return False
            
            # Click start training session
            if not find_and_click(str(IMAGE_PATHS['start_training_session']), description="start training session button"):
                self.logger.error("Could not find start training session button")
                self.should_restart = True
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Error in training setup", e)
            self.should_restart = True
            return False
    
    def _get_progress_value(self, screenshot: np.ndarray) -> Optional[int]:
        """Get progress value from the specified region"""
        try:
            # Extract ROI from screenshot
            roi_image = screenshot[
                PROGRESS_ROI[1]:PROGRESS_ROI[3],
                PROGRESS_ROI[0]:PROGRESS_ROI[2]
            ]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR
            text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
            
            # Look for percentage value
            matches = re.findall(r'(\d+)%', text)
            if matches:
                return int(matches[0])
            
            cv2.imwrite("img/auto_training/roi_image_condition.png", roi_image)
            cv2.imwrite("img/auto_training/roi_image_condition_gray.png", gray)
            return None
            
        except Exception as e:
            self.logger.error("Error getting progress value", e)
            return None
    
    def _get_greens_budget(self, screenshot: np.ndarray) -> Optional[int]:
        """Get current greens budget from the specified region"""
        try:
            # Extract ROI from screenshot
            roi_image = screenshot[
                GREENS_BUDGET_ROI[1]:GREENS_BUDGET_ROI[3],
                GREENS_BUDGET_ROI[0]:GREENS_BUDGET_ROI[2]
            ]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR
            text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
            
            # Look for numeric value
            matches = re.findall(r'\d+', text)
            if matches:
                return int(matches[0])
            
            cv2.imwrite("img/auto_training/roi_image_greens.png", roi_image)
            cv2.imwrite("img/auto_training/roi_image_greens_gray.png", gray)
            return None
            
        except Exception as e:
            self.logger.error("Error getting greens budget", e)
            return None
    
    def _restore_player_condition(self) -> bool:
        """Restore player condition by spending greens"""
        try:
            # Click in the middle of the progress ROI
            roi_center_x = (PROGRESS_ROI[0] + PROGRESS_ROI[2]) // 2
            roi_center_y = (PROGRESS_ROI[1] + PROGRESS_ROI[3]) // 2
            pyautogui.moveTo(roi_center_x, roi_center_y, duration=0.5)
            pyautogui.click()
            time.sleep(CLICK_DELAY)
            time.sleep(2)

            # Find condition selection template
            match = find_on_screen(
                str(IMAGE_PATHS['condition_selection']), 
                description="condition selection"
            )
            if match.center_x is None:
                self.logger.error("Could not find condition selection")
                return False

            # Calculate position for clicking
            click_x = match.center_x + match.width//2 + 30
            click_y = match.center_y + match.height//2 + 30

            # Click 4 times with delay
            for i in range(4):
                pyautogui.moveTo(click_x, click_y, duration=0.5)
                pyautogui.click()
                time.sleep(CLICK_DELAY-0.3)

            # Find and click spend greens button
            if not find_and_click(str(IMAGE_PATHS['spend_greens']), description="spend greens"):
                self.logger.error("Could not find spend greens button")
                return False

            # Find and click exit player menu
            if not find_and_click(str(IMAGE_PATHS['exit_player_menu']), description="exit player menu"):
                self.logger.error("Could not find exit player menu button")
                return False

            return True

        except Exception as e:
            self.logger.error("Error during condition restoration", e)
            return False
    
    def _retry_action(self, action_func: Callable[[], bool], description: str) -> bool:
        """Retry an action multiple times with delay between attempts"""
        for attempt in range(MAX_RECOVERY_ATTEMPTS):
            self.logger.info(f"Attempt {attempt + 1}/{MAX_RECOVERY_ATTEMPTS} for {description}")
            if action_func():
                self.logger.info(f"Successfully completed {description}")
                return True
            if attempt < MAX_RECOVERY_ATTEMPTS - 1:
                self.logger.info(f"Waiting {RECOVERY_DELAY} seconds before next attempt...")
                time.sleep(RECOVERY_DELAY)
        self.logger.error(f"Failed all {MAX_RECOVERY_ATTEMPTS} attempts for {description}")
        return False
    
    def _attempt_recovery(self, failed_action_func: Optional[Callable[[], bool]] = None) -> bool:
        """Attempt to recover from unexpected situations"""
        try:
            # Click center screen
            self.logger.info("Attempting recovery - clicking center screen...")
            def click_center():
                screen_width, screen_height = pyautogui.size()
                pyautogui.moveTo(screen_width//2, screen_height//2, duration=0.5)
                pyautogui.click()
                return True
            
            if not self._retry_action(click_center, "clicking center screen"):
                return False
            
            # Look for continue button
            self.logger.info("Looking for continue button...")
            def click_continue():
                return find_and_click(str(IMAGE_PATHS['continue_button']), description="continue button")
            
            if self._retry_action(click_continue, "clicking continue button"):
                self.logger.info("Found and clicked continue button")
                
                # Retry original action if provided
                if failed_action_func:
                    self.logger.info("Retrying original action...")
                    if self._retry_action(failed_action_func, "original action"):
                        self.logger.info("Original action succeeded after recovery")
                        return True
            
            # Look for training section
            self.logger.info("Looking for training section icon...")
            def click_training():
                return find_and_click(str(IMAGE_PATHS['training_icon']), description="training section icon")
            
            if not self._retry_action(click_training, "clicking training section"):
                self.logger.error("Could not find training section during recovery")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Error during recovery attempt", e)
            return False
    
    def _training_loop(self) -> bool:
        """Execute the main training loop"""
        try:
            while True:
                # Wait and click middle of screen
                self.logger.info("Waiting 2 seconds before middle screen click...")
                time.sleep(2)
                screen_width, screen_height = pyautogui.size()
                pyautogui.moveTo(screen_width//2, screen_height//2, duration=0.5)
                pyautogui.click()
                self.logger.info("Clicked middle of screen")

                # Click repeat training button
                time.sleep(5)
                if not find_and_click(str(IMAGE_PATHS['repeat_training']), description="repeat training button"):
                    self.logger.error("Could not find repeat training button")
                    if not self._attempt_recovery(lambda: find_and_click(str(IMAGE_PATHS['repeat_training']), description="repeat training button")):
                        self.logger.info("Recovery failed")
                        self.should_restart = True
                        return False

                # Check progress and greens
                self.logger.info("Waiting 2 seconds before checking progress...")
                time.sleep(2)
                screenshot = take_screenshot()
                
                # Get progress and greens budget
                progress = self._get_progress_value(screenshot)
                greens = self._get_greens_budget(screenshot)
                
                # Create progress object for logging
                training_progress = TrainingProgress(
                    progress=progress,
                    greens_budget=greens
                )

                if progress is None:
                    self.logger.error("Could not detect progress value")
                    if not self._attempt_recovery(lambda: self._get_progress_value(take_screenshot()) is not None):
                        self.logger.info("Recovery failed")
                        self.should_restart = True
                        return False

                self.logger.info(f"Current training progress: {training_progress.progress}%")

                if progress < MIN_CONDITION_THRESHOLD:
                    self.logger.info(f"Progress is below {MIN_CONDITION_THRESHOLD}%, checking greens budget...")
                    
                    if greens is None:
                        self.logger.error("Could not detect greens budget")
                        self.should_restart = True
                        return False
                        
                    self.logger.info(f"Current greens budget: {training_progress.greens_budget}")
                    
                    # Exit if not enough greens to restore condition
                    if greens < 4:
                        self.logger.warning(f"Not enough greens ({training_progress.greens_budget}) to restore player condition. Exiting training loop.")
                        return True
                    
                    self.logger.info(f"Attempting to restore condition with {training_progress.greens_budget} greens available...")
                    restore_success = self._restore_player_condition()
                    if not restore_success:
                        self.logger.error("Failed to restore player condition")
                        if not self._attempt_recovery(lambda: self._restore_player_condition()):
                            self.logger.info("Recovery failed")
                            self.should_restart = True
                            return False
                    self.logger.info("Successfully restored player condition")

                # Wait before clicking start training
                self.logger.info("Waiting 2 seconds before next training...")
                time.sleep(2)

                # Click start training button
                if not find_and_click(str(IMAGE_PATHS['start_training']), description="start training button"):
                    self.logger.error("Could not find start training button")
                    if not self._attempt_recovery(lambda: find_and_click(str(IMAGE_PATHS['start_training']), description="start training button")):
                        self.logger.info("Recovery failed")
                        self.should_restart = True
                        return False
                        
        except Exception as e:
            self.logger.error("Error in training loop", e)
            self.should_restart = True
            return False
    
    def _prepare_restart(self) -> None:
        """Prepare for bot restart"""
        self.logger.info("Preparing for restart")
        # Exit fullscreen
        pyautogui.press('f11')
        time.sleep(0.5) 
        # Exit the top 11 tab through clicking the x in the tab
        match = find_on_screen(
            str("img/general/top_eleven_tab.jpg"),
            description="top eleven tab"
        )
        self.logger.info(f"Top eleven tab found at: ({match.center_x}, {match.center_y})")
        if match.center_x is not None:
            pyautogui.moveTo(match.center_x + match.width + 10, match.center_y + match.height//2, duration=0.5)
            pyautogui.click()