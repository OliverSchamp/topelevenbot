import time

from typing import Tuple
import numpy as np
from config.penalty_clash_config import (
    CHANGE_EVENT_BUTTON,
    UNSTOPPABLE_CROP,
    UNSTOPPABLE_LOW,
    UNSTOPPABLE_HIGH,
    GREEN_CROP,
    TRIANGLE_CROP,
    GREEN_CHANNEL_THRESHOLD,
    BLUE_CHANNEL_MAX,
    RED_CHANNEL_MAX,
    TRIANGLE_GRAY_THRESHOLD,
    ENERGY_ROI,
    EVENT_ROI,
    CHANGE_EVENT_BUTTON
)
import cv2
from utils.image_processing import fast_click, fast_move, press_only, release_only
from utils.image_processing import take_screenshot_fast as take_screenshot
from config.general_config import ocr_pipeline
from utils.logging_utils import BotLogger

from pathlib import Path

class PenaltyClashBot:
    def __init__(self, team_name: str):
        self.team_name = team_name
        self.logger = BotLogger(__name__)
        template_path = Path("/mnt/SF_NAS/Oliver/tmplt.jpg")
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    def run(self):
        self.logger.info(f"Starting PenaltyClashBot for team: {self.team_name}")
        
        while not ocr_pipeline.run(take_screenshot()).contains_all(["home", "overview"]):
            time.sleep(1)
            self.logger.info("Not in home page yet...")
        
        while True:
            fast_click(CHANGE_EVENT_BUTTON[0], CHANGE_EVENT_BUTTON[1])
            time.sleep(0.5)
            event_screenshot = take_screenshot(EVENT_ROI["x1"], EVENT_ROI["y1"], EVENT_ROI["x2"], EVENT_ROI["y2"])
            event_ocr = ocr_pipeline.run(event_screenshot)
            if event_ocr.contains_all(["penalty", "clash"]):
                fast_click(CHANGE_EVENT_BUTTON[0] + 200, CHANGE_EVENT_BUTTON[1])
                break
        
        for _ in range(2):
            # read the screenshot and click on the play button
            time.sleep(3)
            ocr_result = ocr_pipeline.run(take_screenshot())
            go_coord = ocr_result.get_string_center("PLAY")
            if go_coord is not None:
                fast_click(go_coord[0], go_coord[1])
            else:
                self.logger.warning("Cannot find play button")
                return

        time.sleep(0.01)
        self.logger.info("Moving mouse to (850, 450)")
        fast_move(850, 450)
        self.logger.info("Starting unstoppable/triangle detection loop.")
        try:
            while True:
                self._taking_shot_phase()
                
                # Check if match has ended
                self.logger.info("Checking for 'done' ocr...")
                if ocr_pipeline.run(take_screenshot()).contains_all(["done"]):
                    coord = ocr_result.get_string_center("done")
                    if coord is not None:
                        fast_click(coord[0], coord[1])
                    else:
                        self.logger.warning("Cannot find play button")
                        return
                    self.logger.info("Match ended. Clicked 'done' button.")
                    time.sleep(2)
                    
                    # Check energy with OCR
                    self.logger.info("Checking remaining energy...")
                    screenshot = take_screenshot(ENERGY_ROI['x1'], ENERGY_ROI['y1'], ENERGY_ROI['x2'], ENERGY_ROI['y2'])
                    energy_text = ocr_pipeline.run_recognition(screenshot)
                    
                    try:
                        energy_value = int(energy_text)
                        self.logger.info(f"Energy remaining: {energy_value}")
                        
                        if energy_value == 0:
                            self.logger.info("No energy remaining. Exiting bot.")
                            return
                        else:
                            self.logger.info(f"Energy remaining: {energy_value}. Starting new match.")
                            for _ in range(2):
                                # read the screenshot and click on the play button
                                time.sleep(1.5)
                                ocr_result = ocr_pipeline.run(take_screenshot())
                                go_coord = ocr_result.get_string_center("PLAY")
                                if go_coord is not None:
                                    fast_click(go_coord[0], go_coord[1])
                                else:
                                    self.logger.warning("Cannot find play button")
                                    return
                    except ValueError:
                        self.logger.error(f"Could not parse energy value from text: '{energy_text}'. Exiting.")
                        return
                else:
                    self.logger.debug("No 'done' template found, continuing with current match.")
        except Exception as e:
            self.logger.error("Error in penalty clash loop", error=e)

    def _taking_shot_phase(self):
        self.logger.info("Waiting for unstoppable pixel detection...")
        count = 0
        
        # 1. Start the TOTAL wait timer OUTSIDE the loop
        total_start_time = time.perf_counter()
        
        while True:
            # 2. Start the FRAME processing timer INSIDE the loop
            frame_start_time = time.perf_counter()
            
            cropped = take_screenshot(UNSTOPPABLE_CROP['x1'], UNSTOPPABLE_CROP['y1'], UNSTOPPABLE_CROP['x2'], UNSTOPPABLE_CROP['y2'], mode="GRAY")

            take_screenshot_time = time.perf_counter()

            if self._find_matching_pixel_fast(cropped, UNSTOPPABLE_LOW, UNSTOPPABLE_HIGH):
                # Capture the processing time of the winning frame
                break
            count += 1

            time.sleep(0.005)

        start_press_command = time.perf_counter()    
        press_only()
        press_executed = time.perf_counter()
        # 3. Calculate total duration once the loop breaks
        screenshot_time = take_screenshot_time - frame_start_time
        matching_time = start_press_command - take_screenshot_time
        press_time = press_executed - start_press_command
        total_duration = press_executed - total_start_time
        total_reaction_time = press_executed - frame_start_time
        self.logger.info("Unstoppable pixel detected. Clicked.")
        self.logger.info(f"Unstoppable Stats -> Total wait: {total_duration:.3f}s | Final frame processing: {total_reaction_time:.4f}s, SC {screenshot_time:.4f}s, MATCH {matching_time:.4f}s, PRESS {press_time:.4f}s | Frames checked: {count}")

        try:
            self.logger.info("Waiting for triangle detection in white pixel range...")
            count = 0
            
            # 1. Start the TOTAL wait timer OUTSIDE the loop
            total_start_time = time.perf_counter()
            reaction_time = 0.0 # Fallback in case loop breaks unexpectedly
            leftmost_x = None
            rightmost_x = None
            while True:
                # 2. Start the FRAME processing timer INSIDE the loop
                frame_start_time = time.perf_counter()
                
                green = take_screenshot(GREEN_CROP['x1'], GREEN_CROP['y1'], GREEN_CROP['x2'], GREEN_CROP['y2'])
                
                take_screenshot_time = time.perf_counter()

                # Vectorized green pixel detection
                green_mask = (
                    (green[:, :, 1] > GREEN_CHANNEL_THRESHOLD) &
                    (green[:, :, 2] < BLUE_CHANNEL_MAX) &
                    (green[:, :, 0] < RED_CHANNEL_MAX)
                )
                
                if not np.any(green_mask):
                    count += 1
                    continue
                    
                triangle = take_screenshot(TRIANGLE_CROP['x1'], TRIANGLE_CROP['y1'], TRIANGLE_CROP['x2'], TRIANGLE_CROP['y2'])
                gray = cv2.cvtColor(triangle, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, TRIANGLE_GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    count += 1
                    continue
                elif leftmost_x is None and rightmost_x is None: 
                    green = take_screenshot(GREEN_CROP['x1'], GREEN_CROP['y1'], GREEN_CROP['x2'], GREEN_CROP['y2'])
                
                    take_screenshot_time = time.perf_counter()

                    # Vectorized green pixel detection
                    green_mask = (
                        (green[:, :, 1] > GREEN_CHANNEL_THRESHOLD) &
                        (green[:, :, 2] < BLUE_CHANNEL_MAX) &
                        (green[:, :, 0] < RED_CHANNEL_MAX)
                    )
                    
                    if not np.any(green_mask):
                        count += 1
                        continue                   # Find white pixel range
                    white_pixels = np.column_stack(np.where(green_mask))
                    leftmost_x = np.min(white_pixels[:, 1]) + GREEN_CROP["x1"] - 40
                    rightmost_x = np.max(white_pixels[:, 1]) + GREEN_CROP["x1"]
                    
                    
                cnt = max(contours, key=cv2.contourArea)
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                M = cv2.moments(approx)
                
                if M["m00"] == 0:
                    count += 1
                    continue
                    
                cx = int(M["m10"] / M["m00"]) + TRIANGLE_CROP["x1"]
                self.logger.debug(f"Triangle centroid x: {cx}, white pixel range: {leftmost_x}-{rightmost_x}")
                
                if leftmost_x < cx < rightmost_x:
                    reaction_time = time.perf_counter() - frame_start_time
                    self.logger.info("Triangle detected in white pixel range. Releasing mouse.")
                    break
                elif leftmost_x < cx:
                    reaction_time = time.perf_counter() - frame_start_time
                    self.logger.info("Triangle detected past start of pixel range, but too late released")
                    break
                    
                count += 1

                time.sleep(0.01)
                
        finally:
            # 3. Calculate total duration in the finally block to ensure it always runs
            start_release_command = time.perf_counter()
            release_only()
            release_executed_time = time.perf_counter()
            release_time = release_executed_time - start_release_command
            screenshot_time = take_screenshot_time - frame_start_time
            detect_time = start_release_command - take_screenshot_time
            reaction_time = release_executed_time - frame_start_time
            total_duration = release_executed_time - total_start_time
            self.logger.info("Mouse released.")
            self.logger.info(f"Triangle Stats -> Total wait: {total_duration:.3f}s | Final frame processing: {reaction_time:.4f}s, SC {screenshot_time:.4f}s, DET {detect_time:.4f}s, REL {release_time:.4f}s | Frames checked: {count}")


    # def _taking_shot_phase(self):
    #     self.logger.info("Waiting for unstoppable pixel detection...")
    #     count = 0
    #     while True:
    #         start_screenshot = time.perf_counter()
    #         cropped = take_screenshot(UNSTOPPABLE_CROP['x1'], UNSTOPPABLE_CROP['y1'], UNSTOPPABLE_CROP['x2'], UNSTOPPABLE_CROP['y2'])
    #         # if isinstance(cropped, np.ndarray):
    #         #     img = Image.fromarray(cropped)
    #         # nas_path = Path("/mnt/SF_NAS/Oliver")
    #         # img.save(nas_path / "unstoppable_crop.jpg")

    #         if self._find_matching_pixel_fast(cropped, UNSTOPPABLE_LOW, UNSTOPPABLE_HIGH):
    #             break
 
    #         count += 1
    #     press_only()
    #     self.logger.info("Unstoppable pixel detected. Clicked.")
    #     self.logger.info(f"{time.perf_counter()-start_screenshot}s reaction time")

    #     try:
    #         self.logger.info("Waiting for triangle detection in white pixel range...")
    #         count = 0
    #         while True:
    #             start_screenshot = time.perf_counter()
    #             green = take_screenshot(GREEN_CROP['x1'], GREEN_CROP['y1'], GREEN_CROP['x2'], GREEN_CROP['y2'])
                
    #             # Vectorized green pixel detection - much faster than separate channel operations
    #             green_mask = (
    #                 (green[:, :, 1] > GREEN_CHANNEL_THRESHOLD) &
    #                 (green[:, :, 2] < BLUE_CHANNEL_MAX) &
    #                 (green[:, :, 0] < RED_CHANNEL_MAX)
    #             )
                
    #             # Early exit if no green pixels found
    #             if not np.any(green_mask):
    #                 continue
                    
    #             # Find white pixel range using vectorized operations
    #             white_pixels = np.column_stack(np.where(green_mask))
    #             leftmost_x = np.min(white_pixels[:, 1]) + GREEN_CROP["x1"]
    #             rightmost_x = np.max(white_pixels[:, 1]) + GREEN_CROP["x1"]
                
    #             triangle = take_screenshot(TRIANGLE_CROP['x1'], TRIANGLE_CROP['y1'], TRIANGLE_CROP['x2'], TRIANGLE_CROP['y2'])
    #             gray = cv2.cvtColor(triangle, cv2.COLOR_BGR2GRAY)
    #             _, thresh = cv2.threshold(gray, TRIANGLE_GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
    #             contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             if not contours:
    #                 continue
    #             cnt = max(contours, key=cv2.contourArea)
    #             epsilon = 0.04 * cv2.arcLength(cnt, True)
    #             approx = cv2.approxPolyDP(cnt, epsilon, True)
    #             M = cv2.moments(approx)
    #             if M["m00"] == 0:
    #                 continue
    #             cx = int(M["m10"] / M["m00"]) + TRIANGLE_CROP["x1"]
    #             self.logger.debug(f"Triangle centroid x: {cx}, white pixel range: {leftmost_x}-{rightmost_x}")
    #             if leftmost_x < cx < rightmost_x:
    #                 self.logger.info("Triangle detected in white pixel range. Releasing mouse.")
    #                 break
    #             elif leftmost_x < cx:
    #                 self.logger.info("Triangle detected past start of pixel range, but too late released")
    #                 break
    #             count += 1
    #     finally:
    #         self.logger.info(f"{time.perf_counter() - start_screenshot}s reaction time")
    #         release_only()
    #         self.logger.info("Mouse released.")

    def _find_matching_pixel_fast(self, img: np.ndarray, target_rgb_low: Tuple[int, int, int], target_rgb_high: Tuple[int, int, int]):
        # Vectorized pixel search - much faster than nested loops
        # low = np.array(target_rgb_low)
        # high = np.array(target_rgb_high)
        
        # # Sample every 5th pixel for speed
        # sampled = img[::5, ::5]
        
        # # Vectorized comparison
        # mask = np.all((sampled >= low) & (sampled <= high), axis=2)
        # return np.any(mask)

            # Get template dimensions to calculate the center later
        h, w = self.template.shape


        # 2. Perform the Template Match
        # TM_CCOEFF_NORMED is highly robust to lighting changes while remaining fast
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(img, self.template, cv2.TM_CCOEFF_NORMED)

        # 3. Find the exact pixel coordinate with the highest match confidence
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.84:
            return False

        # 4. Calculate the center of the target
        # max_loc gives the top-left corner of the matched area
        center_x = max_loc[0] + (w // 2)
        center_y = max_loc[1] + (h // 2)

        inside_tolerance = center_x > UNSTOPPABLE_CROP['x1'] + 20 and \
                           center_x < UNSTOPPABLE_CROP['x2'] - 20 and \
                           center_y > UNSTOPPABLE_CROP['y1'] + 20

        return inside_tolerance