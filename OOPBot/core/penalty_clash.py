import time
import pyautogui
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from config.penalty_clash_config import IMAGE_PATHS as PENALTY_IMAGE_PATHS, PENALTY_CLASH_ROI

class PenaltyClashBot:
    def __init__(self, team_name: str):
        self.team_name = team_name
        # Use config values
        self.template_path = PENALTY_IMAGE_PATHS["penalty_clash_template"]
        self.target_rgb = (0, 255, 0)
        self.roi = PENALTY_CLASH_ROI

    def run(self):
        while True:
            # 1. Check for a template
            if self._template_found():
                # 2. Enter "taking shot" phase
                self._taking_shot_phase()
            time.sleep(1)  # Avoid tight loop

    def _template_found(self) -> bool:
        # TODO: Implement template matching using self.template_path
        # For now, always return False (placeholder)
        return False

    def _taking_shot_phase(self):
        # 3. Loop: take screenshot, check every 5th pixel for target RGB
        while True:
            screenshot = self._take_screenshot()
            if self._check_every_5th_pixel(screenshot, self.target_rgb):
                break
            time.sleep(0.05)
        # 4. Click the screen without delay
        pyautogui.click()
        # 5. Loop: take screenshots, crop to ROI, check for triangle shape in thresholded binary image
        while True:
            screenshot = self._take_screenshot()
            roi_img = screenshot.crop(self.roi)
            if self._triangle_in_roi(roi_img):
                break
            time.sleep(0.05)
        # 6. Click the screen without delay, exit taking shot phase
        pyautogui.click()
        # 7. Return to template matching loop

    def _take_screenshot(self) -> Image.Image:
        # Take a screenshot using pyautogui and return as PIL Image
        screenshot = pyautogui.screenshot()
        return screenshot

    def _check_every_5th_pixel(self, img: Image.Image, target_rgb: Tuple[int, int, int]) -> bool:
        arr = np.array(img)
        h, w, _ = arr.shape
        for y in range(0, h, 5):
            for x in range(0, w, 5):
                if tuple(arr[y, x][:3]) == target_rgb:
                    return True
        return False

    def _triangle_in_roi(self, img: Image.Image) -> bool:
        # Convert to numpy array and threshold for (0,255,0)
        arr = np.array(img)
        mask = np.all(arr[:, :, :3] == self.target_rgb, axis=-1)
        # TODO: Implement triangle detection in mask (placeholder: always False)
        return False 