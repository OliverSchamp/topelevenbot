"""
Main bot functionality for Top Eleven
"""

import time
from pathlib import Path
from enum import Enum
import time

from utils.ocr import OCRResult
from utils.logging_utils import BotLogger
from utils.image_processing import find_and_click, find_on_screen, fast_click, take_screenshot
import cv2
import numpy as np
from config.auction_config import IMAGE_PATHS as AUCTION_IMAGE_PATHS
from config.training_config import IMAGE_PATHS as TRAINING_IMAGE_PATHS
from config.ad_config import IMAGE_PATHS as AD_IMAGE_PATHS
from config.general_config import IMAGE_PATHS as GENERAL_IMAGE_PATHS
from config.general_config import MYSTERY_CHOICE_COORDS, ocr_pipeline
from interface import TemplateMatch, ScreenRegion, BotStatus
import subprocess

# Combine all image paths
IMAGE_PATHS = {
    **AUCTION_IMAGE_PATHS,
    **TRAINING_IMAGE_PATHS,
    **AD_IMAGE_PATHS,
    **GENERAL_IMAGE_PATHS
}

class BotMode(Enum):
    """Available bot operation modes"""
    AUCTION = "auction"
    TRAINING = "training"
    AD_WATCH = "ad_watch"
    PENALTY_CLASH = "penalty_clash"

class TopElevenBot:
    """Main bot class for Top Eleven"""
    
    def __init__(self, team_name: str, mode: BotMode = BotMode.TRAINING):
        """Initialize bot"""
        self.team_name = team_name
        self.current_mode = mode
        self.logger = BotLogger(__name__)
        self.should_restart = False
        
        # Verify required images exist
        # self._verify_images()
    
    def _verify_images(self) -> None:
        """Verify that all required images exist"""
        for image_path in IMAGE_PATHS.values():
            if not Path(image_path).exists():
                self.logger.error(f"Missing required image: {image_path}")
                raise FileNotFoundError(f"Missing required image: {image_path}")
    
    def start(self, mode: BotMode) -> None:
        """Start the bot in specified mode"""
        self.current_mode = mode
        self.logger.info(f"Starting bot in {mode.value} mode")
        
        try:
            # Launch game
            self._launch_game()
            
            # Execute selected mode
            if mode == BotMode.AUCTION:
                from core.auction import AuctionBot
                auction_bot = AuctionBot(self.team_name)
                auction_bot.run()
            elif mode == BotMode.TRAINING:
                from core.training import TrainingBot
                training_bot = TrainingBot(self.team_name)
                training_bot.run()
            elif mode == BotMode.AD_WATCH:
                from core.ad_watch import AdWatchBot
                ad_bot = AdWatchBot(self.team_name)
                ad_bot.run()
            elif mode == BotMode.PENALTY_CLASH:
                from core.penalty_clash import PenaltyClashBot
                penalty_clash_bot = PenaltyClashBot(self.team_name)
                penalty_clash_bot.run()
            
        except Exception as e:
            self.logger.error("Error in bot execution", e)
            self.stop()
    
    def stop(self) -> None:
        """Stop the bot and clean up"""
        self.logger.info("Stopping bot")
        self.current_mode = None
        subprocess.run(["sudo", "waydroid", "shell", "input", "keyevent", "3"])
        subprocess.run(["sudo", "waydroid", "shell", "am", "force-stop", "eu.nordeus.topeleven.android"])
    
    def _launch_game(self) -> bool:
        """Assume waydroid is already open. Detect app in top left corner and click."""

        screenshot = cv2.cvtColor(take_screenshot(), cv2.COLOR_BGR2GRAY)
        screenshot_blurred = cv2.medianBlur(screenshot, 5)
        circles = cv2.HoughCircles(screenshot_blurred, 
                            cv2.HOUGH_GRADIENT, 
                            dp=1.2, 
                            minDist=50,
                            param1=40, 
                            param2=18, 
                            minRadius=27, 
                            maxRadius=30)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            top_left_icon = min(circles[0, :], key=lambda c: c[0] + c[1])  
            self.logger.info(f"Top-Left Icon located at: X={top_left_icon[0]}, Y={top_left_icon[1]}")
            fast_click(top_left_icon[0]+(top_left_icon[2]//2), top_left_icon[1]+(top_left_icon[2]//2))

        start_time = time.time()
        while time.time() - start_time < 20:
            screenshot = take_screenshot()
            ocrresult: OCRResult = ocr_pipeline.run(screenshot)
            for textblock in ocrresult.blocks:
                if textblock.text == "HOME":
                    break
                if "DAILY" in textblock.text or "REWARDS" in textblock.text:
                    break

    def get_status(self) -> BotStatus:
        """Get current bot status"""
        return BotStatus(
            mode=self.current_mode.value if self.current_mode else None,
            team_name=self.team_name,
            is_running=self.current_mode is not None
        ) 
    
    def _collect_daily_reward(self):
        
        time.sleep(2)
        
        match = find_on_screen(IMAGE_PATHS["daily_rewards"], description="Daily Rewards Text")

        if match.top_left_x is None:
            return

        if not find_and_click(IMAGE_PATHS["mystery_button"], description="Mystery button"):
            return
        
        time.sleep(1)

        # if not find_and_click(IMAGE_PATHS["mystery_choice"]):
        #     return
        fast_click(MYSTERY_CHOICE_COORDS['x'], MYSTERY_CHOICE_COORDS['y'])
        
        time.sleep(3)
        
        if not find_and_click(IMAGE_PATHS["claim_mystery_choice"], description="Claim mystery choice"):
            return
        
        time.sleep(5)

        return