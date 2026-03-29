"""
Auction bot functionality for Top Eleven
"""

import time
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
from datetime import datetime
from enum import Enum
import pandas as pd

from utils.logging_utils import BotLogger
from utils.image_processing import (
    find_on_screen, 
    safe_int_convert,
    fast_click,
    fast_move
)

from utils.image_processing import take_screenshot
# from utils.image_processing import take_screenshot_fast as take_screenshot
from utils.image_processing import get_fixed_boundaries
from utils.ocr import (
    OCRResultLine,
    TextLine,
    extract_numeric_value,
    OCRResult,
    TextBlock
)
from config.auction_config import (
    DESIRED_POSITIONS,
    MIN_QUALITY,
    MAX_QUALITY,
    MIN_AGE,
    MAX_AGE,
    IMAGE_PATHS,
    MAXIMUM_TOKEN_BUDGET,
    MAXIMUM_MONEY_BUDGET,
    PLAYER_RECORDS_FILE,
    TOTAL_MONEY_AVAILABLE_REGION,
    TOTAL_TOKENS_AVAILABLE_REGION
)

from config.general_config import ocr_pipeline, height, width, mouse_keyboard_controller

from interface import PlayerInfo, TemplateMatch, ScreenRegion, PlayerDetails, PlayerAttributes, BidDetails

class AuctionState(Enum):
    INIT = 1
    SETUP = 2
    TRANSFERS = 3
    BIDDING = 4
    BIDDING_FINISHED = 5
    WIN_MENU = 6
    EXIT = 7

class AuctionResult:
    """Class to represent auction result"""
    SUCCESS = 'success_auction'
    LOST = 'lost_auction'
    ERROR = 'error_during_bidding'
    RESTART_NEEDED = 'restart_needed'
    INSUFFICIENT_TOKENS = 'insufficient_tokens'
    INSUFFICIENT_MONEY = 'insufficient_money'
    EXCEEDS_TOKEN_BUDGET = 'exceeds_token_budget'
    EXCEEDS_MONEY_BUDGET = 'exceeds_money_budget'

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
        self.should_restart: bool = False # if true, transit to exit state
        self.available_tokens = 0.0
        self.available_money = 0.0
        self._last_auction_status = None
        self.state = AuctionState.INIT
        self.tableresult: List[PlayerInfo] = []
        self.won_bidding = False
        self.csv_path = PLAYER_RECORDS_FILE

        # TODO: lower the computational requirements by recording things that don't move position
        self.clock_during_bidding: Optional[TextBlock] = None
        self.clock_during_transfers: Optional[TextBlock] = None
        
        
        self.csv_dataframe = pd.DataFrame() # TODO: record all player details and decisions as a row in the csv
        self.logger.info(f"Initialized auction bot for team: {team_name}")
    
    def set_state(self, new_state):
        self.logger.info(f"{self.state} -> {new_state}")
        self.state = new_state

    def read_screen(self, screenshot: np.ndarray = None) -> OCRResult:
        if screenshot is None:
            screenshot = take_screenshot()
        return ocr_pipeline.run(screenshot)

    def click_word(self, ocr_result: OCRResult, word: str) -> bool:
        word_coord = ocr_result.get_string_center(word)
        if word_coord is not None:
            fast_click(word_coord[0], word_coord[1])
            return True
        else:
            self.logger.warning(f"Cannot find '{word}' button")
            return False

    def _navigate_to_transfers(self) -> bool:
        ocr_result = self.read_screen()
        if not ocr_result.contains_all(["home", "overview"]):
            self.logger.info("Not on home page, waiting...")
            time.sleep(1)
            return False
        
        # crop around total money and total tokens available region, run_recogition and set self.total_money_available and self.total_tokens_available
        total_tokens_crop = take_screenshot(TOTAL_TOKENS_AVAILABLE_REGION['x1'], TOTAL_TOKENS_AVAILABLE_REGION['y1'], TOTAL_TOKENS_AVAILABLE_REGION['x2'], TOTAL_TOKENS_AVAILABLE_REGION['y2'])
        total_money_crop = take_screenshot(TOTAL_MONEY_AVAILABLE_REGION['x1'], TOTAL_MONEY_AVAILABLE_REGION['y1'], TOTAL_MONEY_AVAILABLE_REGION['x2'], TOTAL_MONEY_AVAILABLE_REGION['y2'])
        total_tokens_string = ocr_pipeline.run_recognition(total_tokens_crop)
        total_money_string = ocr_pipeline.run_recognition(total_money_crop)
        self.available_tokens = safe_int_convert(total_tokens_string)
        self.available_money = extract_numeric_value(total_money_string, money=True)
        self.logger.info(f"Available tokens: {self.available_tokens}, Available money: {self.available_money}")
        
        fast_click(height//20, height//20) # click on the home menu
        time.sleep(1)
        ocr_result = self.read_screen()

        out = self.click_word(ocr_result, "transfers")
        time.sleep(2)
        return out

    def on_transfers_page(self, ocr_result: OCRResult) -> bool:
        return ocr_result.contains_all(["transfers", "auctions", "scouting", "assistant"])

    def find_clock(self, ocr_result: OCRResult, mode: str, override: bool = False, screenshot: np.ndarray = None) -> TextBlock:
        if not override:
            if mode == "transfers" and self.clock_during_transfers is not None:
                return self.clock_during_transfers
            if mode == "bidding" and self.clock_during_bidding is not None:
                return self.clock_during_bidding

        clock_block_list = ocr_result.find_blocks_by_regex(r"\d{2}:\d{2}")
        try:
            clock_block = clock_block_list[0]
        except IndexError:
            clock_block_list = ocr_result.find_block_by_string("SOLD", case_sensitive=True)
            try:
                clock_block = clock_block_list[0]

            except Exception as e:
                json_data = ocr_result.model_dump_json(indent=4)
                (Path("debug") / f"ocr_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(json_data)
                # also save screenshot for debugging
                if screenshot is not None:
                    cv2.imwrite(str(Path("debug") / f"findclock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"), cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
                self.logger.error("Clock not found in OCR result, dumping OCR data for debugging")
                raise Exception(f"{e}: Clock not found in OCR result")
        
        if mode == "transfers":
            self.clock_during_transfers = clock_block
        if mode == "bidding":
            self.clock_during_bidding = clock_block

        return clock_block

    def read_transfers_table(self, ocr_result: OCRResult) -> List[PlayerInfo]:
        clock_coords = self.find_clock(ocr_result, mode="transfers", override=True).get_xyxy_coords()
        auction_table_cropped = take_screenshot(0, round(clock_coords[1]-height/21.0), round(clock_coords[0]-width/96.0), height)
        # save auction table image for debugging
        cv2.imwrite(str(Path("debug") / f"auction_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"), cv2.cvtColor(auction_table_cropped, cv2.COLOR_BGR2RGB))
        read_table: OCRResult = ocr_pipeline.run(auction_table_cropped, det_thresh=0.9, rec_thresh=0.4, det_model_idx=1, visualise=True)
        ocr_result_lines: OCRResultLine = read_table.sort_into_lines()

        xs = {}
        ocr_result_lines_full = ocr_result.sort_into_lines()
        for line in ocr_result_lines_full.lines:
            if "pstyl" in line.line_text.lower(): # can use any column header that is unique
                self.logger.info(f"{line.line_text} line found with xs: {line.get_word_xs()}")
                xs = line.get_word_xs()
                break
        boundaries: Dict[str, Tuple[int, int]] = get_fixed_boundaries(points_dict=xs, start_bound=5)

        player_info_list: List[PlayerInfo] = []
        for line in ocr_result_lines.lines:
            sorted_blocks = line.sort_textblocks_into_boundaries(boundaries)
            new_player_info = PlayerInfo()
            for column in sorted_blocks:
                if column.lower() == "name":
                    # for name, if there are multiple blocks, choose the one with the longest text
                    if len(sorted_blocks[column]) > 1:
                        sorted_blocks[column] = sorted(sorted_blocks[column], key=lambda block: len(block.text), reverse=True)
                    new_player_info.name = sorted_blocks[column][0].text
                elif column.lower() == "age":
                    # for age, only set the age if the text is a string that after stripping, represents anumber between 18 and 40, otherwise log and do not set age
                    age_text = sorted_blocks[column][0].text.strip()
                    if age_text.isdigit() and 18 <= int(age_text) <= 40:
                        new_player_info.age = age_text
                elif column.lower() == "qlty":
                    # for qlty, take the first two characters of the leftmost block if multiple blocks
                    if len(sorted_blocks[column]) > 1:
                        sorted_blocks[column] = sorted(sorted_blocks[column], key=lambda block: block.get_xyxy_coords()[0]) # sort by x coordinate
                    new_player_info.qlty = sorted_blocks[column][0].text[:2]
                elif column.lower() == "pstyl":
                    continue
                elif column.lower() == "spec":
                    continue
                elif column.lower() == "deadline":
                    continue
                elif column.lower() == "roles":
                    for block in sorted_blocks[column]:
                        new_player_info.roles.append(block.text)
                elif column.lower() == "value":
                    new_player_info.value = sorted_blocks[column][0].text
                else:
                    self.logger.info(f"Unrecognized column in auction table, skipping: {column}")
            
            new_player_info.clean_roles()

            min_y, max_y = line.get_line_y_min_y_max()
            for k, v in boundaries.items():
                if k.lower() == "pstyl":
                    pstyl_box = [v[0], min_y, v[1], max_y]
                    playstyle_cropped = auction_table_cropped[pstyl_box[1]:pstyl_box[3], pstyl_box[0]:pstyl_box[2]]
                    gray = cv2.cvtColor(playstyle_cropped, cv2.COLOR_BGR2GRAY)
                    new_player_info.pstyl = (gray < 250).any()
                if k.lower() == "spec":
                    spec_box = [v[0], min_y, v[1], max_y]
                    special_cropped = auction_table_cropped[spec_box[1]:spec_box[3], spec_box[0]:spec_box[2]]
                    gray = cv2.cvtColor(special_cropped, cv2.COLOR_BGR2GRAY)
                    new_player_info.spec = (gray < 250).any()

            player_info_list.append(new_player_info)


        for player_info in player_info_list:
            self.logger.info(f"PLAYER: {player_info.to_string()}")

        return player_info_list

    def age_in_spec(self, player_info: PlayerInfo) -> Tuple[bool, str]:
        return safe_int_convert(player_info.age) <= MAX_AGE, "age not in spec"

    def everything_in_spec(self, player_info: PlayerInfo) -> Tuple[bool, str]:
        try:
            if not safe_int_convert(player_info.age) <= MAX_AGE:
                return False, "age too high"

            if not safe_int_convert(player_info.age) >= MIN_AGE:
                return False, "age too low"
            # TODO: fix role reading. isolate role blocks and force ocr output.
            # only get people with a single role
            if len(player_info.roles) == 1:
                if player_info.roles[0] not in DESIRED_POSITIONS:
                    return False, "role not in desired roles"
            else:
                return False, "more than one role"
            
            # playstyle, must have
            if not player_info.pstyl:
                return False, "has no playstyle"
            
            # check quality
            if safe_int_convert(player_info.qlty) > MAX_QUALITY:
                return False, "quality above max"

            if safe_int_convert(player_info.qlty) < MIN_QUALITY:
                return False, "quality below min"

            return True, ""
        except Exception as e:
            self.logger.error(f"Error in everything_in_spec: {str(e)}")
            return False, "error in spec evaluation"

    def start_bidding(self, row_info: PlayerInfo):
        self.logger.info(f"Starting bidding for player: {row_info.name}, roles: {row_info.roles}, age: {row_info.age}, quality: {row_info.qlty}, playstyle: {row_info.pstyl}")
        if self.click_word(ocr_pipeline.run(take_screenshot()), row_info.name):
            time.sleep(3) # some delay when opening player menu

        bid_start = True
        while True:
            bidding_active = self.bidding_active()
            if bidding_active is True:
                self.logger.info("Bidding active, checking if winning and if in budget")
                # TODO: you actually only need to find the clock once, then you always know where it is for bidding and for table
                screenshot = take_screenshot()
                if self.is_winning(self.find_clock(self.read_screen(screenshot), mode="bidding", screenshot=screenshot).get_xyxy_coords()) and not bid_start:
                    self.logger.info("Currently winning the auction, waiting...")
                    continue
                else:
                    self.logger.info("Not winning the auction/ just entered page, checking if in budget to place bid")
                    bid_start = False
                    in_budget, message = self.is_in_budget()
                    if message != "":
                        self.logger.info(f"Not in budget, reason: {message}")

                    if in_budget:
                        self.logger.info("Not winning but still in budget, placing bid")
                        self.click_word(ocr_pipeline.run(take_screenshot()), "bid")
                    else:
                        self.logger.info("Not winning and not in budget, exiting bidding")
                        self.won_bidding = False
                        self.set_state(AuctionState.BIDDING_FINISHED)
                        self.exit_bidding()
                        break

            elif bidding_active is False: # bidding no longer active and have not exited, means have won
                self.logger.info("Bidding no longer active, should have won")
                self.won_bidding = True
                self.set_state(AuctionState.BIDDING_FINISHED)
                break
            
            else: # error somewhere
                self.logger.info(f"Error determining if bidding is active, exiting bidding to be safe. Bidding active: {bidding_active}")
                self.set_state(AuctionState.EXIT)
                break


    def reread_textblock(self, textblock: TextBlock) -> TextBlock:
        textblock_xyxy = textblock.get_xyxy_coords()
        textblock_crop = take_screenshot(textblock_xyxy[0], textblock_xyxy[1], textblock_xyxy[2], textblock_xyxy[3])
        textblock.text = ocr_pipeline.run_recognition(textblock_crop)
        return textblock

    def bidding_active(self):
        screenshot = take_screenshot()
        ocr_result = self.read_screen()
        if self.clock_during_bidding is None:
            self.find_clock(ocr_result, mode="bidding", screenshot=screenshot)
        # save to debug
        screenshot_pil = Image.fromarray(screenshot)
        screenshot_pil.save(Path("debug") / f"bidding_screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        next_offer_present = ocr_result.contains_all(["next", "offer"])
        # fast localized read of specific clock location
        clock_active = False
        if ":" in self.reread_textblock(self.clock_during_bidding).text:
            clock_active = True
        
        return next_offer_present and clock_active
    
    def detect_green(self, img: np.ndarray) -> bool:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixel_count = cv2.countNonZero(mask)
        return green_pixel_count > 100

    def is_winning(self, clock_location: Tuple[int, int, int, int]) -> bool:
        clock_screenshot = take_screenshot(clock_location[0], clock_location[1], clock_location[2], clock_location[3])
        is_green = self.detect_green(clock_screenshot)
        return is_green

    def is_in_budget(self) -> Tuple[bool, str]:
        """Template match for the token and the dollar sign
            Crop the regions to the right of the matches (hardccode)
            Read these regions with ocr and continue"""

        token_location = find_on_screen(IMAGE_PATHS['token_icon']).get_xyxy()
        money_location = find_on_screen(IMAGE_PATHS['money_icon']).get_xyxy()
        token_crop = [token_location[2], token_location[1], money_location[0], token_location[3]] # hardcoded, this should not really change
        money_crop = [money_location[2], money_location[1], money_location[2] + (width//15), money_location[3]] # hardcoded, this should not really change
        
        screenshot = take_screenshot()
        token_crop_img = screenshot[token_crop[1]:token_crop[3], token_crop[0]-5:token_crop[2]]
        money_crop_img = screenshot[money_crop[1]:money_crop[3], money_crop[0]-5:money_crop[2]]
        
        token_string = ocr_pipeline.run_recognition(token_crop_img)
        money_string = ocr_pipeline.run_recognition(money_crop_img)
        # SAVE TOKEN AND MONEY CROPS FOR DEBUGGING
        # cv2.imwrite(str(Path("debug") / f"token_crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"), token_crop_img)
        # cv2.imwrite(str(Path("debug") / f"money_crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"), money_crop_img)

        if safe_int_convert(token_string) > MAXIMUM_TOKEN_BUDGET:
            return False, "voluntary exit (tokens)"
        
        if extract_numeric_value(money_string, money=True) > MAXIMUM_MONEY_BUDGET:
            return False, "voluntary exit (money)"
        
        if safe_int_convert(token_string) > self.available_tokens:
            return False, "outbid (tokens)"
        
        if extract_numeric_value(money_string, money=True) > self.available_money:
            return False, "outbid (money)"

        return True, ""

    def escape_winning_message(self) -> bool:
        """press esc every few seconds until detect transfers page again"""

        for _ in range(10):
            if self.on_transfers_page(self.read_screen()):
                return True

            time.sleep(3)
            mouse_keyboard_controller.press_key("esc")
            time.sleep(3)
        
        return False

    def read_clock_n_wait(self, ocr_result: OCRResult):
        if len(self.tableresult) == 0:
            self.logger.info("No players read from table, likely OCR error, waiting and trying again")
            time.sleep(10) # wait some time before trying again
            return
        player_name = self.tableresult[0].name
        player_name_2 = self.tableresult[1].name
        player_name_3 = self.tableresult[2].name
        if not ocr_result.contains_all([player_name], case_sensitive=False) and not ocr_result.contains_all([player_name_2], case_sensitive=False) and not ocr_result.contains_all([player_name_3], case_sensitive=False):
            self.logger.info(f"Player names {player_name}, {player_name_2}, and {player_name_3}  not found on screen, likely auction table refreshed, skipping clock reading and waiting")
            time.sleep(10) # wait some time before trying again
            return        
        
        clock_text = self.find_clock(ocr_result, mode="transfers", override=True).text

        if ':' not in clock_text:
            self.logger.info("Clock text does not contain ':'")
            return

        clock_parts = clock_text.split(':')
        if len(clock_parts) >= 2:
            minutes = safe_int_convert(clock_parts[0])
            seconds = safe_int_convert(clock_parts[1])
            if minutes is not None and seconds is not None:
                clock_seconds = minutes * 60 + seconds
                self.logger.info(f"Waiting for {clock_seconds} seconds until next auction")
        else:
            self.logger.error(f"Could not parse clock time: {clock_text}, defaulting to 90 seconds")
            clock_seconds = 90

        time.sleep(clock_seconds)

    def exit_bidding(self) -> bool:
        mouse_keyboard_controller.press_key("esc")
        time.sleep(2)
        return self.on_transfers_page(self.read_screen())
    
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
                        self.set_state(AuctionState.EXIT)
                        return
                    continue
                
                while not self.should_restart:
                    # wait endlessly for transfers page
                    ocr_result = ocr_pipeline.run(take_screenshot())
                    if self.on_transfers_page(ocr_result):
                        self.click_word(ocr_result, "auctions")
                        time.sleep(1)
                        self.state =  AuctionState.TRANSFERS
                    else:
                        self.set_state(AuctionState.EXIT)
                        break
                    
                    ocr_result = ocr_pipeline.run(take_screenshot())
                    self.click_word(ocr_result, "NAME")
                    time.sleep(0.5)
                    self.click_word(ocr_result, "AGE")
                    time.sleep(0.5)
                    # move mouse away
                    fast_move(height//20, height//20)
                    time.sleep(0.1)
                    screenshot = take_screenshot()
                    # save image
                    # cv2.imwrite(str(Path("debug") / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"), screenshot)
                    ocr_result = ocr_pipeline.run(screenshot)
                    self.tableresult: List[PlayerInfo] = self.read_transfers_table(ocr_result)
                    for row in self.tableresult:
                        if not row.is_valid():
                            self.logger.info(f"INVALID: {row.to_string()}")
                            continue

                        age_in_spec, out_message = self.age_in_spec(row)
                        everything_in_spec, out_message = self.everything_in_spec(row)
                        self.logger.info(f"IN SPEC: {row.name}, AGE: {age_in_spec}, EVERYTHING: {everything_in_spec}, REASON: {out_message}")

                        if age_in_spec and not everything_in_spec:
                            if row != self.tableresult[-1]:  # Adjust the threshold as needed
                                continue
                            else:
                                self.logger.info("At bottom of table, waiting and breaking")
                                self.read_clock_n_wait(ocr_pipeline.run(take_screenshot()))
                                break
                        elif age_in_spec and everything_in_spec:
                            self.set_state(AuctionState.BIDDING)
                            self.start_bidding(row)
                        else:
                            self.read_clock_n_wait(ocr_pipeline.run(take_screenshot()))
                            break

                        if self.won_bidding == False:
                            continue

                        self.escape_winning_message()

        except Exception as e:
            self.logger.error("Critical error in auction bot execution", exc_info=True)
            self.set_state(AuctionState.EXIT)