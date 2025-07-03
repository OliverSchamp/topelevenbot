"""
Training bot functionality for Top Eleven
"""

import time
import pyautogui
import cv2
import numpy as np
import pytesseract
import re
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, List

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
    IMAGE_PATHS,
    HEADERS_AND_COORDS,
    DRILL_SCROLL_AMOUNT
)
from interface import TemplateMatch, ScreenRegion, TrainingProgress

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# TODO: still bad behaviour when a player improves a star. does restart game though....
# TODO: check condition from the start of training also

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
        
        # Load drills CSV
        try:
            csv_path = Path("fast_trainer_sheet/drills_per_position.csv")
            self.drills_df = pd.read_csv(csv_path)
            self.logger.info("Successfully loaded drills configuration")
        except Exception as e:
            self.logger.error("Failed to load drills configuration", e)
            self.drills_df = None
    
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
            
            time.sleep(3)
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
                CONFIDENCE_THRESHOLD-0.1, 
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
    
    def _get_player_position(self) -> Optional[str]:
        """Extract player position from the screen using OCR"""
        try:
            # Find confirm player template
            confirm_match = find_on_screen(
                str(IMAGE_PATHS['confirm_player']),
                CONFIDENCE_THRESHOLD,
                "confirm checkbox"
            )
            
            # Find player template
            player_match = find_on_screen(
                str(IMAGE_PATHS['player_to_train']),
                CONFIDENCE_THRESHOLD+0.1,
                "player"
            )
            
            if confirm_match.center_x is None or player_match.center_x is None:
                self.logger.error("Could not find required templates for position extraction")
                return None
                
            # Define ROI coordinates
            x1 = confirm_match.top_left_x + confirm_match.width  # Right of confirm template
            y1 = player_match.top_left_y  # Top of confirm template
            x2 = player_match.top_left_x  # Left of player template
            y2 = player_match.top_left_y + player_match.height  # Bottom of player template
            
            # Take screenshot and crop ROI
            screenshot = take_screenshot()
            roi = np.array(screenshot)[y1:y2, x1:x2]
            #save roi image
            cv2.imwrite("img/auto_training/roi_image_position.png", roi)
            
            thresh = cv2.inRange(roi, (50, 50, 50), (90, 90, 90))
            # save thresh image
            cv2.imwrite("img/auto_training/roi_image_position_thresh.png", thresh)
            text = pytesseract.image_to_string(thresh).strip().upper()
            self.logger.info(f"Extracted position text: {text}")
            
            return text
            
        except Exception as e:
            self.logger.error("Error extracting player position", e)
            return None

    def _get_required_drills(self, position: str) -> Optional[Dict[str, int]]:
        """Get required drills for position from loaded DataFrame"""
        try:
            if self.drills_df is None:
                self.logger.error("Drills configuration not loaded")
                return None
                
            # Find the row for this position
            position_row = self.drills_df[self.drills_df['Position'] == position]
            if position_row.empty:
                self.logger.error(f"Position {position} not found in configuration")
                return None
                
            # Get non-zero columns and their values
            drills = {}
            for column in self.drills_df.columns[1:]:  # Skip 'Position' column
                value = position_row[column].iloc[0]
                if value > 0:
                    drills[column] = value
                    
            self.logger.info(f"Required drills for {position}: {drills}")
            return drills
            
        except Exception as e:
            self.logger.error("Error getting required drills", e)
            return None

    def _find_drill_by_ocr(self, target_drills: List[str]) -> Tuple[List[Tuple[int, int, str]], Optional[int]]:
        """
        Find drill using OCR on detected regions
        Args:
            target_drills: List of drill names to search for
        Returns:
            Tuple containing:
            - Optional tuple of (x,y,name) coordinates and name of found drill
            - Optional x coordinate for scrolling (middle point between two drills)
        """
        try:
            # Take screenshot
            screenshot = take_screenshot()
            gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get image with only black and white
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Store all valid drill regions
            drill_regions = []
            target_drill_info = []
            
            # First pass: collect all valid regions and find target drill
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter for rectangle-like shapes (aspect ratio and size)
                if w > 300 and h > 100 and y < 500:
                    drill_regions.append((x, y, w, h))
                    
                    # Extract ROI for OCR
                    x1 = x + int(w*0.05)
                    y1 = y + int(h*0.45)
                    x2 = x1 + int(w*0.9)
                    y2 = y1 + int(50)
                    
                    roi = gray[y1:y2, x1:x2]
                    
                    # Apply OCR
                    text = pytesseract.image_to_string(roi).strip().upper()
                    self.logger.info(f"OCR text: {text}")
                    
                    # Check if text matches any target drill
                    for drill_name in target_drills:
                        if drill_name.upper() in text:
                            center_x = x1 + (x2 - x1) // 2
                            center_y = y1 + (y2 - y1) // 2
                            target_drill_info.append((center_x, center_y, drill_name))
                            self.logger.info(f"Found drill: {drill_name} at ({center_x}, {center_y})")
                            break
            
            # Sort regions by x coordinate
            drill_regions.sort(key=lambda r: r[0])
            
            # Find suitable scroll position (middle point between two drills)
            scroll_x = None
            if len(drill_regions) >= 2:
                largest_gap = 0
                gap_center = None
                
                for i in range(len(drill_regions) - 1):
                    current_drill = drill_regions[i]
                    next_drill = drill_regions[i + 1]
                    gap = next_drill[0] - (current_drill[0] + current_drill[2])
                    
                    if gap > largest_gap:
                        largest_gap = gap
                        gap_center = current_drill[0] + current_drill[2] + gap // 2
                
                scroll_x = gap_center
            
            return target_drill_info, scroll_x
            
        except Exception as e:
            self.logger.error("Error finding drill with OCR", e)
            return None, None

    def _search_and_drag_drills_to_slot(self, remaining_drills: Dict[str, int], header_name: str) -> bool:
        """Drag drills to empty slots based on requirements
        
        should drag left 5 times looking for required drills. If all drills are found, return true.

        Returns false if there are still drills left to find
        """
        self.logger.info("Looking for required drills...")
        
        for attempt in range(5):
            # Check if empty slot exists and exit if there are no more slots to fill
            slot_template_match = find_on_screen(
                str(IMAGE_PATHS['empty_slot']), 
                CONFIDENCE_THRESHOLD-0.2, 
                "empty slot"
            )
            
            if slot_template_match.confidence < CONFIDENCE_THRESHOLD-0.2:
                self.logger.info("No more empty slots found")
                break
                
            # Try to find any of the remaining drills
            target_drills = [name for name, count in remaining_drills.items() if count > 0]
 
            drill_info_list, scroll_x = self._find_drill_by_ocr(target_drills)
            
            for drill_info in drill_info_list:
                drill_x, drill_y, drill_name = drill_info
                slot_x = slot_template_match.center_x
                slot_y = slot_template_match.center_y

                if drill_name in target_drills:
                    target_drills.remove(drill_name)
                    for _ in range(remaining_drills[drill_name]):
                        # take a screenshot and search for an empty slot
                        empty_slot_match = find_on_screen(
                            str(IMAGE_PATHS['empty_slot']), 
                            CONFIDENCE_THRESHOLD-0.2, 
                            "empty slot"
                        )
                        if empty_slot_match.confidence < CONFIDENCE_THRESHOLD-0.2:
                            self.logger.error("No more empty slots found")
                            return True
                        
                        slot_x = empty_slot_match.center_x
                        slot_y = empty_slot_match.center_y

                        self.logger.info(f"Dragging {drill_name} from ({drill_x}, {drill_y}) to ({slot_x}, {slot_y})")
                        pyautogui.moveTo(drill_x, drill_y, duration=0.5)
                        pyautogui.mouseDown()
                        time.sleep(0.1)
                        pyautogui.moveTo(drill_x, drill_y+(slot_y-drill_y)/2, duration=DRAG_DURATION//10)
                        pyautogui.moveTo(slot_x, slot_y, duration=DRAG_DURATION//10)
                        time.sleep(0.1)
                        pyautogui.mouseUp()
                        time.sleep(CLICK_DELAY//10)
                    
                    remaining_drills[drill_name] = 0
                    self.logger.info(f"Remaining drills: {remaining_drills}")
                    # if no more remaining drills, break
                    if all(count == 0 for count in remaining_drills.values()):
                        return True
                
            # If drill not found and we have a scroll position, scroll left
            _, screen_height = pyautogui.size()
            center_y = screen_height // 2
            if scroll_x:
                self.logger.info(f"Scrolling from x position: {scroll_x}")
                pyautogui.moveTo(scroll_x, center_y, duration=0.5)
                pyautogui.mouseDown()
                time.sleep(0.1)
                pyautogui.moveTo(scroll_x - DRILL_SCROLL_AMOUNT, center_y, duration=0.5)
                pyautogui.mouseUp()
                time.sleep(0.5)
            else:
                self.logger.error("No suitable scroll position found")
                self.should_restart = True
                return False
                
            # if no more attempts and still remaining drills, return false
            if attempt == 4 and any(count > 0 for count in remaining_drills.values()):
                return False
        
        return True

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
            
            # Get player position
            position = self._get_player_position()
            if not position:
                self.logger.error("Could not determine player position")
                self.should_restart = True
                return False
            
            # Get required drills for position
            required_drills = self._get_required_drills(position)
            if not required_drills:
                self.logger.error("Could not determine required drills")
                self.should_restart = True
                return False
            
            # Click confirm
            if not find_and_click(str(IMAGE_PATHS['confirm']), description="confirm button"):
                self.logger.error("Could not find confirm button")
                self.should_restart = True
                return False
            
            time.sleep(1)
            
            # Click drills
            if not find_and_click(str(IMAGE_PATHS['drills']), description="drills button"):
                self.logger.error("Could not find drills button")
                self.should_restart = True
                return False
            
            time.sleep(1)
            
            # Click physical and mental using hardcoded coordinates
            all_drills_found = False
            for header_name, coords in HEADERS_AND_COORDS.items():
                pyautogui.moveTo(coords[0], coords[1], duration=0.5)
                pyautogui.click()
                time.sleep(CLICK_DELAY)
            
                # Fill empty slots with required drills
                all_drills_found = self._search_and_drag_drills_to_slot(required_drills, header_name)
                
                if all_drills_found:
                    break
            
            if not all_drills_found:
                self.logger.error("Failed to drag all required drills")
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
                self.logger.info("Waiting 4 seconds before checking progress...")
                time.sleep(4)
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
                    # TODO: Recovery is bugged
                    # if not self._attempt_recovery(lambda: self._get_progress_value(take_screenshot()) is not None):
                    #     self.logger.info("Recovery failed")
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
            pyautogui.moveTo(match.top_left_x + match.width + 10, match.center_y, duration=0.5)
            pyautogui.click()