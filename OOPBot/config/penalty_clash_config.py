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
UNSTOPPABLE_CROP = {"x1": 162, "y1": 76, "x2": 794, "y2": 325}
UNSTOPPABLE_LOW = (40, 250, 5)
UNSTOPPABLE_HIGH = (50, 255, 15)

# Green/white pixel detection crop and thresholds
GREEN_CROP = {"x1": 550 // 2, "y1": 750 // 2, "x2": 1350 // 2, "y2": 850 // 2}
TRIANGLE_CROP = {"x1": 550 // 2, "y1": 765 // 2, "x2": 1350 // 2, "y2": 785 // 2}
GREEN_CHANNEL_THRESHOLD = 230
BLUE_CHANNEL_MAX = 70
RED_CHANNEL_MAX = 130

TRIANGLE_GRAY_THRESHOLD = 230

ENERGY_ROI = {"x1": 141 // 2, "y1": 87 // 2, "x2": 167 // 2, "y2": 889 // 2}


EVENT_ROI = {"x1": 170, "y1": 81, "x2": 950, "y2": 235}
CHANGE_EVENT_BUTTON = (187, 158)