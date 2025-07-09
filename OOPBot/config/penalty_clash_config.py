from pathlib import Path

IMAGE_PATHS = {
    "home": "img/general/home_menu.jpg", 
    "events": "img/penalty_clash/events.jpg", 
    "go": "img/penalty_clash/go.jpg", 
    "play": "img/penalty_clash/play.jpg",
    "done": "img/penalty_clash/done.jpg",
}

PENALTY_CLASH_ROI = (100, 100, 200, 200)  # Placeholder ROI (left, top, right, bottom)

# Unstoppable detection crop and thresholds
UNSTOPPABLE_CROP = {"x1": 250, "y1": 120, "x2": 1650, "y2": 600}
UNSTOPPABLE_LOW = (40, 250, 5)
UNSTOPPABLE_HIGH = (50, 255, 15)

# Green/white pixel detection crop and thresholds
GREEN_CROP = {"x1": 550, "y1": 750, "x2": 1350, "y2": 850}
TRIANGLE_CROP = {"x1": 550, "y1": 765, "x2": 1350, "y2": 785}
GREEN_CHANNEL_THRESHOLD = 230
BLUE_CHANNEL_MAX = 70
RED_CHANNEL_MAX = 130

TRIANGLE_GRAY_THRESHOLD = 230

ENERGY_ROI = {"x1": 141, "y1": 87, "x2": 167, "y2": 889}