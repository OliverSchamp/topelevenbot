"""
Configuration settings for the auction functionality
"""

from pathlib import Path

# Timing and Delays
CLICK_DELAY = 2.0
AUCTION_CHECK_INTERVAL = 1.0  # seconds between auction status checks
MAX_WIN_MESSAGE_ATTEMPTS = 3  # number of times to look for win message

# Screen Regions
ROW_HEIGHT = 86
FIRST_PLACE_BOX = {
    'x1': 638, 
    'y1': 525,
    'x2': 1046,
    'y2': 569
}

QUALITY_BOX = {
    'x1': 543,
    'y1': 164,
    'x2': 606,
    'y2': 225
}

POSITION_BOXES = [
    {'x1': 1210, 'y1': 270, 'x2': 1285, 'y2': 310},  # Primary
    {'x1': 1305, 'y1': 270, 'x2': 1380, 'y2': 310},  # Secondary
    {'x1': 1400, 'y1': 270, 'x2': 1475, 'y2': 310}   # Tertiary
]

# Playstyle coordinates
PLAYSTYLE_TEXT_REGION = {
    'x1': 750,
    'y1': 652,
    'x2': 1427,
    'y2': 715
}

# Image Recognition
CONFIDENCE_THRESHOLD = 0.8

# Bidding
MAXIMUM_TOKEN_BUDGET = 20
MAXIMUM_MONEY_BUDGET = 5 #M

# ROIs for the tokens and money that I have left
TOTAL_TOKENS_AVAILABLE_REGION = {
    'x1': 953,
    'y1': 18,
    'x2': 1046,
    'y2': 64
}
TOTAL_MONEY_AVAILABLE_REGION = {
    'x1': 1766,
    'y1': 19,
    'x2': 1862,
    'y2': 61
}

AUCTION_WON_PIXEL = {'x': 1455, 'y': 562}

AUCTION_WON_COLOR = {'r': 65, 'g': 185, 'b': 55}

# Valid positions in the game
VALID_POSITIONS = [
    "GK",   # Goalkeeper
    "DC",   # Center Back
    "DL",   # Left Back
    "DR",   # Right Back
    "DMC",  # Defensive Midfielder
    "MC",   # Center Midfielder
    "ML",   # Left Midfielder
    "MR",   # Right Midfielder
    "AMC",  # Attacking Midfielder
    "AML",  # Left Winger
    "AMR",  # Right Winger
    "ST"    # Striker
]

# Player filtering settings
DESIRED_POSITIONS = ["DC", "DMC", "DR"]  # Only bid on players in these positions
MIN_QUALITY = 0  # Minimum quality percentage to consider
MAX_QUALITY = 100  # Maximum quality percentage to consider
MIN_AGE = 18  # Minimum age to consider
MAX_AGE = 19  # Maximum age to consider
EXPECTED_VALUE_MULTIPLIER = 1.0

# File Paths
IMAGE_PATHS = {
    'ldplayer': Path("img/general/ldplayer_icon.jpg"),
    'ldplayer_open': Path("img/general/ldplayer_open.jpg"),
    'home_menu': Path("img/general/home_menu.jpg"),
    'transfers': Path("img/auto_auction/transfers.jpg"),
    'age': Path("img/auto_auction/age.jpg"),
    'top_of_clock': Path("img/auto_auction/top_of_clock.jpg"),
    'value': Path("img/auto_auction/value.jpg"),
    'auctions': Path("img/auto_auction/auctions.jpg"),
    'bid': Path("img/auto_auction/bid.jpg"),
    'exit_bidding': Path("img/auto_auction/exit_bidding.jpg"),
    'name': Path("img/auto_auction/name.jpg"),
    'next_offer': Path("img/auto_auction/next_offer.jpg"),
    'next_offer_moneys': Path("img/auto_auction/next_offer_moneys.jpg"),
    'exit_win_message': Path("img/auto_auction/exit_win_message.jpg"),
    'winning_offer': Path("img/auto_auction/winning_offer.jpg"),
    'playstyles': Path("img/auto_auction/playstyle.jpg"),
    'offers': Path("img/auto_auction/offers.jpg")
}

# OCR Configuration
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Record Keeping
PLAYER_RECORDS_FILE = Path('player_records.csv')
FAST_TRAINERS_FILE = Path("fast_trainer_sheet/fast_trainers.csv") 