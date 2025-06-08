"""
Main entry point for Top Eleven Bot
"""

import sys
import time
from pathlib import Path
from utils.logging_utils import setup_logging
from core.bot import TopElevenBot, BotMode
from utils.image_processing import find_and_click
from config.auction_config import IMAGE_PATHS

def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    
    # Create necessary directories
    Path("img/auto_auction").mkdir(parents=True, exist_ok=True)
    Path("img/player_positions").mkdir(parents=True, exist_ok=True)
    Path("img/auto_ads").mkdir(parents=True, exist_ok=True)
    Path("img/x_examples").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize bot with team name
    bot = TopElevenBot("FC 69")  # Replace with your team name
    
    # Get bot mode from user input
    print("\nSelect bot mode:")
    print("1. Auction")
    print("2. Training")
    print("3. Ad Watch")
    
    while True:
        try:
            mode = int(input("\nEnter mode number (1-3): "))
            if mode == 1:
                selected_mode = BotMode.AUCTION
                break
            elif mode == 2:
                selected_mode = BotMode.TRAINING
                break
            elif mode == 3:
                selected_mode = BotMode.AD_WATCH
                break
            else:
                print("Invalid mode number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 3.")
    
    while True:
        try:
            # Start bot in selected mode
            bot.start(selected_mode)
            
            # If we get here, the bot needs to restart
            print("\nRestarting bot in 3 seconds...")
            time.sleep(3)
            
            # Try to restart Top Eleven
            if find_and_click(str("img/general/top_eleven.jpg"), description="restart top eleven"):
                time.sleep(20)  # Wait for game to load
            
        except KeyboardInterrupt:
            print("\nBot stopped by user")
            bot.stop()
            break
        except Exception as e:
            print(f"Critical error: {e}")
            bot.stop()
            sys.exit(1)

if __name__ == "__main__":
    main() 