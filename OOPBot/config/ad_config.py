"""
Configuration settings for the ad watching functionality
"""

from pathlib import Path

# Timing and Delays
CLICK_DELAY = 1.0  # Delay between clicks in seconds
AD_CHECK_INTERVAL = 0.2  # How often to check for X button (in seconds)

# Image Recognition
CONFIDENCE_THRESHOLD = 0.9  # Minimum confidence score for template matching

# Image Paths
IMAGE_PATHS = {
    'ldplayer': Path("img/general/ldplayer_icon.jpg"),
    'ldplayer_open': Path("img/general/ldplayer_open.jpg"),
    'green_hud': Path("img/auto_ads/greens_hud.jpg"),
    'greens_ads_button': Path("img/auto_ads/greens_ads_button.jpg"),
}

# X button templates directory
X_BUTTONS_DIR = Path("img/x_examples") 

MAX_TIME_WITHOUT_X = 120  # 2 minutes in seconds