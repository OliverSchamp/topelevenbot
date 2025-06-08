"""
Auction bot functionality for Top Eleven
"""

import time
import json
import pandas as pd
import pyautogui
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2
from datetime import datetime

from utils.logging_utils import BotLogger
from utils.image_processing import (
    find_and_click, 
    find_on_screen, 
    take_screenshot,
    clock_image_preprocessing,
    safe_int_convert
)
from utils.ocr import (
    extract_text_from_region,
    extract_numeric_value,
    get_player_name
)
from config.auction_config import (
    CLICK_DELAY,
    ROW_HEIGHT,
    FIRST_PLACE_BOX,
    QUALITY_BOX,
    POSITION_BOXES,
    VALID_POSITIONS,
    DESIRED_POSITIONS,
    MIN_QUALITY,
    MAX_QUALITY,
    MIN_AGE,
    MAX_AGE,
    IMAGE_PATHS,
    MAXIMUM_TOKEN_BUDGET,
    PLAYER_RECORDS_FILE,
    FAST_TRAINERS_FILE,
    PLAYSTYLE_TEXT_REGION,
    CONFIDENCE_THRESHOLD
)
from interface import TemplateMatch, ScreenRegion, PlayerDetails, PlayerAttributes, AuctionPageResult

class AuctionResult:
    """Class to represent auction result"""
    SUCCESS = 'success'
    LOST = 'lost'
    SKIPPED = 'skipped'
    ERROR = 'error'
    RESTART_NEEDED = 'restart_needed'

class ComparisonResult:
    """Class to represent value comparison results"""
    TOO_OLD = 'too_old'
    TOO_YOUNG = 'too_young'
    OUTSIDE_QUALITY_RANGE = 'outside_quality_range'
    VALUE_HIGHER = 'value_higher'
    VALUE_EQUAL = 'value_equal'
    VALUE_LOWER = 'value_lower'
    WRONG_PLAYSTYLE = "wrong_playstyle"

class AuctionBot:
    """Bot for handling player auctions"""
    
    def __init__(self, team_name: str):
        """Initialize auction bot"""
        self.team_name = team_name
        self.logger = BotLogger(__name__)
        self.evaluated_players: set = set()
        self.should_restart: bool = False
        self.logger.info(f"Initialized auction bot for team: {team_name}")
        
        # Load fast trainers dataset
        try:
            self.fast_trainers_df = pd.read_csv(FAST_TRAINERS_FILE)
        except Exception as e:
            self.logger.error(f"Error loading fast trainers data: {str(e)}")
            raise
    
    def run(self) -> None:
        """Main auction bot loop"""
        try:
            self.logger.info("Starting auction bot main loop")
            while True:
                self.should_restart = False
                
                # Navigate to transfers
                self.logger.info("Attempting to navigate to transfers section")
                if not self._navigate_to_transfers():
                    if self.should_restart:
                        self.logger.info("Restart needed after navigation attempt")
                        self._prepare_restart()
                        return
                    continue
                
                # Load fast trainers data
                self.logger.info("Loading fast trainers data")
                df_fast_trainers = self.fast_trainers_df
                if df_fast_trainers is None:
                    self.logger.error("Failed to load fast trainers data, preparing for restart")
                    self._prepare_restart()
                    return
                
                while not self.should_restart:
                    # Start from the first row position
                    self.logger.info("Getting initial Y position for player scanning")
                    current_y = self._get_initial_y_position()
                    if current_y is None:
                        if self.should_restart:
                            self.logger.info("Restart needed after getting initial position")
                            self._prepare_restart()
                            return
                        break
                    
                    self.logger.info(f"Starting player evaluation from Y position: {current_y}")
                    while not self.should_restart:
                        player_attrs, status, new_y = self._process_auction_page(current_y, df_fast_trainers)
                        
                        # Always save player record if we have attributes
                        if player_attrs:
                            save_player_record(player_attrs)
                        
                        # Handle processing status
                        if not status:
                            self.logger.error("Error processing auction page, preparing for restart")
                            self._prepare_restart()
                            return
                        
                        # Handle new Y position
                        if new_y is not None:
                            current_y = new_y
                        else:
                            current_y += ROW_HEIGHT
                            time.sleep(1)
                
                if self.should_restart:
                    self.logger.info("Restart flag set, preparing for restart")
                    self._prepare_restart()
                    return
                    
        except Exception as e:
            self.logger.error("Critical error in auction bot execution", exc_info=True)
            self._prepare_restart()
    
    def _prepare_restart(self) -> None:
        """Prepare for restart by exiting fullscreen"""
        try:
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
                time.sleep(0.1)
                pyautogui.click()
        except Exception as e:
            self.logger.error("Error preparing for restart", e)
    
    def _navigate_to_transfers(self) -> bool:
        """Navigate to transfers section"""
        try:
            # Find and click transfers
            if not find_and_click(str(IMAGE_PATHS['transfers']), description="transfers"):
                self.logger.error("Could not find transfers button")
                self.should_restart = True
                return False

            time.sleep(2)

            # Find and click auctions if needed
            if not find_and_click(str(IMAGE_PATHS['age']), description="age"):
                if not find_and_click(str(IMAGE_PATHS['auctions']), description="auctions"):
                    self.logger.error("Could not find auctions")
                    self.should_restart = True
                    return False

                time.sleep(0.3)
                
                # Reorder players by name then age
                if not self._reorder_players():
                    self.should_restart = True
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error("Error navigating to transfers", e)
            self.should_restart = True
            return False
    
    def _get_initial_y_position(self) -> Optional[int]:
        """Get initial Y position for scanning players"""
        try:
            match = find_on_screen(
                str(IMAGE_PATHS['top_of_clock']), 
                description="top of clock"
            )
            if match.center_y is None:
                self.logger.error("Could not find top of clock")
                return None
            return match.center_y
            
        except Exception as e:
            self.logger.error("Error getting initial Y position", e)
            return None
    
    def _reorder_players(self) -> bool:
        """Reorder players by clicking name then age columns"""
        try:
            # Click name column to temporarily sort by name
            if not find_and_click(str(IMAGE_PATHS['name']), description="name column"):
                self.logger.error("Could not find name column")
                return False
            time.sleep(CLICK_DELAY)
            
            # Click age column to sort by age
            if not find_and_click(str(IMAGE_PATHS['age']), description="age column"):
                self.logger.error("Could not find age column")
                return False
            time.sleep(CLICK_DELAY)
            
            return True
            
        except Exception as e:
            self.logger.error("Error reordering players", e)
            return False
        
    def _handle_new_auction(self) -> Optional[int]:
        """Handle waiting for new auction and return new y position"""
        try:
            screenshot = take_screenshot()
            success = self._wait_for_new_auctions(screenshot)  # removed current_y parameter
            if success:
                return self._get_initial_y_position()
            return None
        except Exception as e:
            self.logger.error("Error handling new auction", e)
            return None
    
    def _process_auction_page(self, current_y: int, df_fast_trainers: pd.DataFrame) -> tuple[PlayerAttributes, bool, Optional[int]]:
        """Process a single player on the auction page
        Returns:
            tuple containing:
            - PlayerAttributes: Player data (always returned, even if incomplete)
            - bool: Status (True if processing succeeded, False if error occurred)
            - Optional[int]: New Y position if auction was reset, None otherwise
        """
        # Initialize player attributes at the start
        player_attrs = PlayerAttributes()
        
        try:
            screenshot = take_screenshot()
            
            # Check if we've reached bottom of page
            if current_y > pyautogui.size()[1] - ROW_HEIGHT:
                self.logger.info(f"Reached bottom of page at Y: {current_y}, waiting for new auctions")
                exit_bidding = self._exit_bidding()
                new_y = self._handle_new_auction()
                return player_attrs, exit_bidding, new_y
            
            # Get player details
            player_details = self._get_player_details(screenshot, current_y)
            if player_details is None:
                player_attrs.reason_rejected = "Failed to get player details"
                return player_attrs, False, None
            
            # Update basic attributes
            name, age, value, quality, positions, playstyle = player_details
            player_attrs.name = name
            player_attrs.age = age
            player_attrs.value = value
            player_attrs.quality = quality
            player_attrs.positions = positions
            player_attrs.playstyle = playstyle
            
            # Skip if already evaluated
            if name in self.evaluated_players:
                self.logger.info(f"Player already evaluated: {name}")
                player_attrs.reason_rejected = "Player already evaluated"
                return player_attrs, self._exit_bidding(name), None
            
            # Check age range
            if age < MIN_AGE:
                player_attrs.reason_rejected = f"Player too young: {age}"
                return player_attrs, self._exit_bidding(name), None
            elif age > MAX_AGE:
                self.logger.info(f"Player too old: {age}")
                player_attrs.reason_rejected = f"Player too old: {age}"
                exit_bidding = self._exit_bidding(name)
                new_y = self._handle_new_auction()
                return player_attrs, exit_bidding, new_y
            
            # Check quality range
            if quality < MIN_QUALITY or quality > MAX_QUALITY:
                self.logger.info(f"Quality outside range: {quality}")
                player_attrs.reason_rejected = f"Quality outside range: {quality}"
                return player_attrs, self._exit_bidding(name), None
            
            # Check positions
            if not any(pos in DESIRED_POSITIONS for pos in positions if pos):
                self.logger.info(f"No desired positions: {positions}")
                player_attrs.reason_rejected = f"No desired positions: {positions}"
                return player_attrs, self._exit_bidding(name), None
            
            # Get expected value for age
            age_col = f"{age}yo"
            try:
                quality_row = df_fast_trainers[df_fast_trainers['%'] == str(quality)+'%']
                expected_value = float(quality_row[age_col].iloc[0])
                player_attrs.expected_value = expected_value
            except KeyError:
                player_attrs.reason_rejected = f"Age {age} not in dataset"
                return player_attrs, self._exit_bidding(name), None
            
            # Compare values
            if value < expected_value:
                player_attrs.comparison_result = ComparisonResult.VALUE_LOWER
                player_attrs.reason_rejected = "Value lower than expected"
                return player_attrs, self._exit_bidding(name), None
            elif value == expected_value:
                player_attrs.comparison_result = ComparisonResult.VALUE_EQUAL
            elif value > expected_value:
                player_attrs.comparison_result = ComparisonResult.VALUE_HIGHER
            else:
                self.logger.error(f"Value {value} cannot be compared to expected value {expected_value}")
                player_attrs.comparison_result = ComparisonResult.VALUE_LOWER
                player_attrs.reason_rejected = "Could not compare value to expected value"
                return player_attrs, self._exit_bidding(name), None

            # Place bid
            bid_result, bid_amount = self._handle_bidding()
            player_attrs.bid_amount = bid_amount
            if bid_result == AuctionResult.ERROR:
                player_attrs.reason_rejected = "Error during bidding"
                return player_attrs, False, None
            elif bid_result == AuctionResult.SKIPPED:
                player_attrs.reason_rejected = "Bidding skipped"
                return player_attrs, self._exit_bidding(name), None
            
            # Bid was placed
            player_attrs.was_bid_placed = True
            
            return player_attrs, self._exit_bidding(name), None
            
        except Exception as e:
            self.logger.error(f"Error processing auction page: {str(e)}")
            player_attrs.reason_rejected = f"Exception occurred: {str(e)}"
            return player_attrs, False, new_y
    
    def _wait_for_new_auctions(self, screenshot: np.ndarray) -> bool:
        """Wait for new auctions to appear"""
        try:
            clock_template_match = find_on_screen(
                str(IMAGE_PATHS['top_of_clock']), 
                description="top of clock"
            )
            clock_x = clock_template_match.top_left_x
            clock_y = clock_template_match.top_left_y
            clock_w = clock_template_match.width

            # Extract wait time
            region = screenshot[clock_y:clock_y+2*ROW_HEIGHT, clock_x:clock_x+clock_w]
            region = clock_image_preprocessing(region)
            wait_start_time = extract_text_from_region(region, preprocess=False)
            
            # Parse wait time
            wait_parts = wait_start_time.split(' ')
            if len(wait_parts) >= 2:
                minutes = safe_int_convert(wait_parts[0])
                seconds = safe_int_convert(wait_parts[1])
                if minutes is not None and seconds is not None:
                    wait_time = minutes * 60 + seconds
                    self.logger.info(f"Waiting for {wait_time} seconds")
                    time.sleep(wait_time + 2)
                    return True
            
            self.logger.error(f"Could not parse wait time: {wait_start_time}")
            return False
            
        except Exception as e:
            self.logger.error("Error waiting for new auctions", e)
            return False
    
    def _get_player_details(self, screenshot: np.ndarray, current_y: int) -> Optional[Tuple[str, int, float, int, List[Optional[str]], Optional[str]]]:
        """Get all details for a player"""
        try:
            # Get column coordinates
            name_template_match = find_on_screen(
                str(IMAGE_PATHS['name']),
                description="name column"
            )
            name_x = name_template_match.top_left_x
            name_center_x = name_template_match.center_x
            name_center_y = name_template_match.center_y
            name_w = name_template_match.width

            age_template_match = find_on_screen(
                str(IMAGE_PATHS['age']),
                description="age column"
            )
            age_x = age_template_match.top_left_x
            age_w = age_template_match.width

            value_template_match = find_on_screen(
                str(IMAGE_PATHS['value']),
                description="value column"
            )
            value_x = value_template_match.top_left_x
            value_w = value_template_match.width
            
            if any(x is None for x in [name_x, age_x, value_x]):
                return None
            
            # Extract player details
            name_region = screenshot[current_y:current_y+ROW_HEIGHT, name_x+85:name_x+name_w] # 85 hardcoded to remove flag
            player_name = get_player_name(extract_text_from_region(name_region))
            
            age_region = screenshot[current_y:current_y+ROW_HEIGHT, age_x:age_x+age_w]
            age_text = extract_text_from_region(age_region)
            age = int(''.join(filter(str.isdigit, age_text)))
            
            value_region = screenshot[current_y:current_y+ROW_HEIGHT, value_x:value_x+value_w]
            value_text = extract_text_from_region(value_region)
            value = extract_numeric_value(value_text)
            
            pyautogui.moveTo(name_center_x, current_y + ROW_HEIGHT//2, duration=0.5)
            pyautogui.click()
            time.sleep(CLICK_DELAY)
            
            # Get quality
            quality = self._get_player_quality()
            
            # Get positions
            positions = self._get_player_positions()
            
            # Get playstyle
            playstyle = self._get_player_playstyle()
            if playstyle is None:
                self.logger.error("Could not determine playstyle, issue here")
            if playstyle == '':
                self.logger.error("Player should have no playstyle")

            return player_name, age, value, quality, positions, playstyle
            
        except Exception as e:
            self.logger.error("Error getting player details", e)
            return None
    
    def _get_player_quality(self) -> Optional[int]:
        """Get player quality from quality box"""
        try:
            screenshot = take_screenshot()
            quality_region = screenshot[
                QUALITY_BOX['y1']:QUALITY_BOX['y2'],
                QUALITY_BOX['x1']:QUALITY_BOX['x2']
            ]
            quality_text = extract_text_from_region(quality_region)
            return int(''.join(filter(str.isdigit, quality_text)))
        except Exception as e:
            self.logger.error("Error getting player quality", e)
            return None
    
    def _get_player_positions(self) -> List[Optional[str]]:
        """Get player positions"""
        positions = []
        screenshot = take_screenshot()
        
        for i, box in enumerate(POSITION_BOXES):
            try:
                position_region = screenshot[
                    box['y1']:box['y2'],
                    box['x1']:box['x2']
                ]
                position_text = extract_text_from_region(position_region)
                positions.append(self._get_closest_position(position_text))
            except Exception as e:
                self.logger.error(f"Error getting position {i}", e)
                positions.append(None)
        
        return positions
    
    def _get_closest_position(self, detected_text: str) -> Optional[str]:
        """Match detected text to the closest valid position"""
        if not detected_text:
            return None
            
        detected_text = detected_text.strip().upper()
        
        if detected_text in VALID_POSITIONS:
            return detected_text
            
        # Simple character-based similarity score
        def similarity_score(pos1: str, pos2: str) -> float:
            pos1, pos2 = pos1.upper(), pos2.upper()
            matches = sum(1 for a, b in zip(pos1, pos2) if a == b)
            length_diff = abs(len(pos1) - len(pos2))
            return matches - length_diff * 0.5
        
        best_match = None
        best_score = float('-inf')
        
        for valid_pos in VALID_POSITIONS:
            score = similarity_score(detected_text, valid_pos)
            if score > best_score:
                best_score = score
                best_match = valid_pos
        
        return best_match
    
    def _compare_player_value(self, age: int, quality: int, value: float, df: pd.DataFrame) -> str:
        """Compare the player's value with expected value from CSV data"""
        try:
            # Check age range
            if age > MAX_AGE:
                self.logger.info(f"Age {age} exceeds maximum age {MAX_AGE}, stopping auction processing")
                return ComparisonResult.TOO_OLD
            if age < MIN_AGE:
                self.logger.info(f"Age {age} below minimum age {MIN_AGE}, skipping player")
                return ComparisonResult.TOO_YOUNG
            
            # Check quality range
            if quality < MIN_QUALITY or quality > MAX_QUALITY:
                self.logger.info(f"Quality {quality}% is outside desired range ({MIN_QUALITY}%-{MAX_QUALITY}%), skipping player")
                return ComparisonResult.OUTSIDE_QUALITY_RANGE
            
            quality_row = df[df['%'] == str(quality)+'%']
            
            if quality_row.empty:
                self.logger.error(f"Could not find quality {quality}% in the dataset")
                return ComparisonResult.VALUE_LOWER
                
            age_col = f"{age}yo"
            try:
                expected_value = float(quality_row[age_col].iloc[0])
            except KeyError:
                self.logger.error(f"Could not find age {age} in the dataset, automatically rejecting player")
                return ComparisonResult.TOO_OLD
            self.logger.info(f"Value comparison - Expected: {expected_value}M, Actual: {value}M")
            
            if abs(value - expected_value) < 0.01:
                self.logger.info("Value matches expected value")
                return ComparisonResult.VALUE_EQUAL
            elif value > expected_value:
                self.logger.info("Value higher than expected")
                return ComparisonResult.VALUE_HIGHER
            else:
                self.logger.info("Value lower than expected")
                return ComparisonResult.VALUE_LOWER
                
        except Exception as e:
            self.logger.error("Error comparing values", exc_info=True)
            return ComparisonResult.VALUE_LOWER
    
    def _handle_bidding(self) -> Tuple[str, Optional[float]]:
        """Handle bidding process"""
        try:
            # Check starting bid amount
            self.logger.info("Getting starting bid amount")
            starting_bid = self._get_next_offer_amount()
            if starting_bid is None:
                self.logger.error("Could not determine starting bid amount")
                return AuctionResult.ERROR, None
                
            if starting_bid > MAXIMUM_TOKEN_BUDGET:
                self.logger.info(f"Starting bid {starting_bid} exceeds maximum budget {MAXIMUM_TOKEN_BUDGET}")
                return AuctionResult.SKIPPED, starting_bid
            
            # Initial bid
            self.logger.info(f"Placing initial bid of {starting_bid} tokens")
            if not find_and_click(str(IMAGE_PATHS['bid']), description="bid button"):
                self.logger.error("Could not find bid button for initial bid")
                return AuctionResult.ERROR, None
            
            current_bid = starting_bid
            self.logger.info("Starting auction monitoring loop")
            while True:
                status = self._monitor_auction_status()
                self.logger.info(f"Current auction status: {status}")
                
                if status == 'won':
                    self.logger.info(f"Won auction with final bid of {current_bid} tokens")
                    return AuctionResult.SUCCESS, current_bid
                elif status == 'lost':
                    self.logger.info(f"Lost auction at bid of {current_bid} tokens")
                    return AuctionResult.LOST, current_bid
                elif status == 'restart_needed':
                    self.logger.warning("Restart needed during bidding")
                    self.should_restart = True
                    return AuctionResult.RESTART_NEEDED, current_bid
                elif status == 'outbid':
                    self.logger.info("Outbid, checking next offer amount")
                    next_amount = self._get_next_offer_amount()
                    if next_amount is None:
                        self.logger.error("Could not determine next bid amount")
                        return AuctionResult.LOST, current_bid
                    elif next_amount > MAXIMUM_TOKEN_BUDGET:
                        self.logger.info(f"Next bid amount {next_amount} exceeds budget {MAXIMUM_TOKEN_BUDGET}")
                        return AuctionResult.LOST, current_bid
                    
                    self.logger.info(f"Placing new bid of {next_amount} tokens")
                    if not find_and_click(str(IMAGE_PATHS['bid']), description="bid button"):
                        self.logger.error("Could not find bid button for next bid")
                        return AuctionResult.ERROR, None
                    current_bid = next_amount
                
                time.sleep(CLICK_DELAY)
                
        except Exception as e:
            self.logger.error("Error in bidding process", exc_info=True)
            return AuctionResult.ERROR, None
    
    def _get_next_offer_amount(self) -> Optional[float]:
        """Get the amount needed for the next offer"""
        try:
            screenshot = take_screenshot()
            
            # Find the next offer display
            self.logger.debug("Looking for next offer display")
            next_offer_template_match = find_on_screen(
                str(IMAGE_PATHS['next_offer']),
                description="next offer"
            )
            next_offer_x = next_offer_template_match.top_left_x
            next_offer_y = next_offer_template_match.top_left_y
            next_offer_w = next_offer_template_match.width
            next_offer_h = next_offer_template_match.height
            
            if any(x is None for x in [next_offer_x, next_offer_y, next_offer_w, next_offer_h]):
                self.logger.error("Could not find next offer display")
                return None
            
            # Next offer region is 60 pixels to the right
            next_offer_region = screenshot[
                next_offer_y:next_offer_y+next_offer_h,
                next_offer_x+next_offer_w:next_offer_x+next_offer_w+60
            ]
            
            # Preprocess the image
            next_offer_region = cv2.inRange(next_offer_region, (230, 230, 230), (255, 255, 255))
            next_offer_text = extract_text_from_region(next_offer_region, preprocess=False)
            
            amount = extract_numeric_value(next_offer_text)
            if amount is None:
                self.logger.error(f"Could not extract numeric value from text: {next_offer_text}")
                return None
                
            self.logger.info(f"Next offer amount: {amount} tokens")
            return amount
            
        except Exception as e:
            self.logger.error("Error getting next offer amount", exc_info=True)
            return None
    
    def _monitor_auction_status(self) -> str:
        """Monitor the current auction status"""
        try:
            screenshot = take_screenshot()
            
            # Check if auction ended
            next_offer_template_match = find_on_screen(
                str(IMAGE_PATHS['next_offer']),
                description="next offer"
            )
            next_offer_x = next_offer_template_match.top_left_x
            
            
            if next_offer_x is None:
                self.logger.info("Next offer display not found, checking auction end conditions")

                # Check for winning offer template
                winning_offer_template_match = find_on_screen(
                    str(IMAGE_PATHS['winning_offer']),
                    description="winning offer message"
                )
                winning_offer_x = winning_offer_template_match.top_left_x

                if winning_offer_x is not None:
                    self.logger.info("Found winning offer message, auction finished")
                    congratulations_template_match = find_on_screen(
                        str(IMAGE_PATHS['congratulations']),
                        description="congratulations message"
                    )
                    congratulations_x = congratulations_template_match.top_left_x
                    if congratulations_x is not None:
                        self.logger.info("Found congratulations message, auction won")
                    else:
                        self.logger.info("Could not find congratulations message, auction lost")
                        return 'lost'

                self.logger.info("Waiting 5 seconds before checking win message")
                time.sleep(5)              # find and click the exit win message
                if find_and_click(str(IMAGE_PATHS['exit_win_message']), description="exit win message"):
                    self.logger.info("Found and clicked win message")
                    return 'won'
                
                # If we can't find any of the expected messages, we need to restart
                self.logger.error("Cannot find win/loss messages - restart needed")
                return 'restart_needed'

            # Check if still in first place
            self.logger.debug("Checking first place status")
            first_place_region = screenshot[
                FIRST_PLACE_BOX['y1']:FIRST_PLACE_BOX['y2'],
                FIRST_PLACE_BOX['x1']:FIRST_PLACE_BOX['x2']
            ]
            first_place_text = extract_text_from_region(first_place_region)
            
            if self.team_name not in first_place_text:
                self.logger.info(f"Team {self.team_name} no longer in first place")
                return 'outbid'
            
            self.logger.debug("Still in first place")
            return 'ongoing'
            
        except Exception as e:
            self.logger.error("Error monitoring auction status", exc_info=True)
            return 'error'
    
    def _exit_bidding(self, name: Optional[str] = None) -> bool:
        """Exit the bidding screen"""
        if name is not None:
            self.evaluated_players.add(name)
        return find_and_click(str(IMAGE_PATHS['exit_bidding']), description="exit bidding button")
    
    def _get_player_playstyle(self) -> Optional[str]:
        """Get player playstyle if available"""
        try:
            # Find and click playstyles button
            self.logger.debug("Looking for playstyles button")
            if not find_and_click(str(IMAGE_PATHS['playstyles']), description="playstyles button"):
                self.logger.info("No playstyles button found - player has no playstyle")
                return None
            
            # Take screenshot and extract playstyle text
            self.logger.debug("Extracting playstyle text")
            screenshot = take_screenshot()
            playstyle_region = screenshot[
                PLAYSTYLE_TEXT_REGION['y1']:PLAYSTYLE_TEXT_REGION['y2'],
                PLAYSTYLE_TEXT_REGION['x1']:PLAYSTYLE_TEXT_REGION['x2']
            ]
            playstyle_text = extract_text_from_region(playstyle_region)

            self.logger.info(f"Found playstyle: {playstyle_text}")
            
            # Return to bidding menu
            self.logger.debug("Returning to bidding menu")
            if not find_and_click(str(IMAGE_PATHS['offers']), description="offers button"):
                self.logger.error("Could not find offers button to return to bidding")
                return None
                
            
            return playstyle_text.strip()
            
        except Exception as e:
            self.logger.error("Error getting player playstyle", exc_info=True)
            return None

def save_player_record(player: PlayerAttributes) -> None:
    """Save player record to CSV file"""
    try:
        # Convert player model to dict
        player_dict = player.model_dump()
        
        # Convert timestamp to string
        player_dict['timestamp'] = player_dict['timestamp'].isoformat()
        
        # Convert positions list to string
        player_dict['positions'] = ','.join(filter(None, player_dict['positions']))
        
        # Create DataFrame
        df = pd.DataFrame([player_dict])
        
        # Append to CSV
        file_exists = Path(PLAYER_RECORDS_FILE).exists()
        df.to_csv(PLAYER_RECORDS_FILE, mode='a', header=not file_exists, index=False)
        
    except Exception as e:
        logger = BotLogger(__name__)
        logger.error(f"Error saving player record: {str(e)}") 