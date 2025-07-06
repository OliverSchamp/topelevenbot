"""
Main bot functionality for Top Eleven
"""

import time
import pyautogui
from pathlib import Path
from typing import Optional
from enum import Enum
import logging

from utils.logging_utils import BotLogger
from utils.image_processing import find_and_click, find_on_screen
from config.auction_config import IMAGE_PATHS as AUCTION_IMAGE_PATHS
from config.training_config import IMAGE_PATHS as TRAINING_IMAGE_PATHS
from config.ad_config import IMAGE_PATHS as AD_IMAGE_PATHS
from config.general_config import IMAGE_PATHS as GENERAL_IMAGE_PATHS
from config.general_config import MYSTERY_CHOICE_COORDS
from interface import TemplateMatch, ScreenRegion, BotStatus

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
        
        # Initialize pyautogui safety
        pyautogui.FAILSAFE = True
        
        # Verify required images exist
        self._verify_images()
    
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
    
    def _launch_game(self) -> bool:
        """Launch the game through LDPlayer"""
        try:
            # Check if LDPlayer is already open
            match = find_on_screen(
                str(IMAGE_PATHS['ldplayer_open']), 
                description="LDPlayer window"
            )
            
            # If LDPlayer is not already open, try to launch it
            if match.center_x is None:
                self.logger.info("LDPlayer not found, attempting to launch it")
                if not find_and_click(str(IMAGE_PATHS['ldplayer']), description="LDPlayer icon"):
                    self.logger.error("Could not find LDPlayer icon")
                    return False
            else:
                self.logger.info("LDPlayer already running")
                
            # Enter fullscreen
            pyautogui.press('f11')
            time.sleep(0.3)

            #if has been restarted
            if find_and_click(str("img/general/top_eleven.jpg"), description="restart top eleven"):
                time.sleep(20)  # Wait for game to load

            # check for daily rewards popup and select a daily reward
            self._collect_daily_reward()

            # click at [0, 500]
            pyautogui.moveTo(0, 500, duration=0.5)
            pyautogui.click()
            time.sleep(0.5)
            
            # Find and click home menu if bot is not in ad watch mode
            if self.current_mode != BotMode.AD_WATCH:
                if not find_and_click(str(IMAGE_PATHS['home_menu']), description="home menu"):
                    self.logger.error("Could not find home menu")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error("Error launching game", e)
            return False
    
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
        pyautogui.moveTo(MYSTERY_CHOICE_COORDS['x'], MYSTERY_CHOICE_COORDS['y'], duration=0.5)
        pyautogui.click()
        
        time.sleep(3)
        
        if not find_and_click(IMAGE_PATHS["claim_mystery_choice"], description="Claim mystery choice"):
            return
        
        time.sleep(5)

        return