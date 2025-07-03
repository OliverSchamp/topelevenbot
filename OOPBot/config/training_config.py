"""
Configuration settings for the training functionality
"""

from pathlib import Path

# Timing and Delays
CLICK_DELAY = 1.0  # Delay between clicks in seconds
DRAG_DURATION = 0.25  # Duration for drag operations
SCROLL_AMOUNT = -100  # Negative value scrolls down
MAX_SCROLL_ATTEMPTS = 20  # Maximum number of scroll attempts to find player

# Image Recognition
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence score for template matching
PROMO_CONFIDENCE_THRESHOLD = 0.7  # Lower threshold for promo button

# Progress Monitoring
PROGRESS_ROI = [1441, 571, 1660, 626]  # Region of interest for progress monitoring
GREENS_BUDGET_ROI = [1171, 20, 1234, 59]  # Region of interest for greens budget monitoring. TODO: check how many greens are left before recovering player condition
MIN_CONDITION_THRESHOLD = 30  # Minimum condition percentage before restoration

# Recovery Settings
MAX_RECOVERY_ATTEMPTS = 3  # Maximum number of recovery attempts
RECOVERY_DELAY = 5  # Delay between recovery attempts

# OCR Configuration
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

HEADERS_AND_COORDS = {'ATTACK': [250, 160], 'DEFENSE': [610, 160], 'POSSESSION': [970, 160], 'PHYSICAL AND MENTAL': [1330, 160]}
DRILL_SCROLL_AMOUNT = 500

# Image Paths
IMAGE_PATHS = {
    'ldplayer': Path("img/general/ldplayer_icon.jpg"),
    'home_menu': Path("img/general/home_menu.jpg"),
    'promo_x': Path("img/general/promo_x_1.jpg"),
    'training': Path("img/auto_training/training_section.jpg"),
    'players': Path("img/auto_training/players.jpg"),
    'player_to_train': Path("img/auto_training/players/s_leca.jpg"),
    'confirm_player': Path("img/auto_training/confirm_player.jpg"),
    'confirm': Path("img/auto_training/confirm.jpg"),
    'drills': Path("img/auto_training/drills.jpg"),
    'empty_slot': Path("img/auto_training/empty_train_slot.jpg"),
    'start_training': Path("img/auto_training/start_training.jpg"),
    'start_training_session': Path("img/auto_training/start_training_session.jpg"),
    'repeat_training': Path("img/auto_training/repeat_training.jpg"),
    'condition_selection': Path("img/auto_training/condition_selection.jpg"),
    'spend_greens': Path("img/auto_training/spend_greens.jpg"),
    'exit_player_menu': Path("img/auto_training/exit_player_menu.jpg"),
    'continue_button': Path("img/auto_training/continue.jpg"),
    'home_section': Path("img/auto_training/home_section.jpg"),
    'training_icon': Path("img/auto_training/training_section_icon.jpg"),
    'ldplayer_open': Path("img/general/ldplayer_open.jpg")
} 