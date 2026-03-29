"""
OCR utility functions for text extraction from images
"""

import cv2
import re
import logging
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import openvino as ov
import pyclipper
import math
from shapely.geometry import Polygon
from pydantic import BaseModel, Field
import re
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from copy import deepcopy


logger = logging.getLogger(__name__)

class TextBlock(BaseModel):
    """Represents a single detected and recognized line of text."""
    box: List[Tuple[float, float]] = Field(description="List of 4 (x, y) coordinates: [top-left, top-right, bottom-right, bottom-left]")
    text: str = Field(description="The recognized text string")
    det_score: float = Field(description="Confidence score of the text detection bounding box")
    rec_score: float = Field(description="Confidence score of the text recognition")

    def get_centre_coord(self) -> Tuple[int, int]:
        return (int((self.box[2][0]+self.box[0][0]) // 2), int((self.box[2][1]+self.box[0][1]) // 2))

    def get_xyxy_coords(self) -> Tuple[int, int, int, int]:
        x_coords = [pt[0] for pt in self.box]
        y_coords = [pt[1] for pt in self.box]
        return (int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords)))

class TextLine(BaseModel):
    """Represents a full horizontal line of text."""
    line_text: str = Field(description="The combined text of all words in this line")
    words: List[TextBlock] = Field(description="The individual text blocks, ordered left-to-right")

    def get_line_y(self) -> float:
        """Returns the average y-coordinate of the line based on its words."""
        if not self.words:
            return 0.0
        return sum(word.get_centre_coord()[1] for word in self.words) / len(self.words)
    
    def get_line_y_min_y_max(self) -> Tuple[float, float]:
        """Returns the minimum and maximum y-coordinates of the line based on its words."""
        if not self.words:
            return (0.0, 0.0)
        y_coords_tl = [word.get_xyxy_coords()[1] for word in self.words]
        y_coords_br = [word.get_xyxy_coords()[3] for word in self.words]
        return (min(y_coords_tl), max(y_coords_br))

    def get_word_xs(self) -> Dict[str, float]:
        """Returns a dictionary mapping each word to its x-coordinate."""
        return {word.text: word.get_centre_coord()[0] for word in self.words}

    def sort_textblocks_into_boundaries(self, boundaries: Dict[str, Tuple[int, int]]) -> Dict[str, List[TextBlock]]:
        """Sorts the line's text blocks into the provided x-coordinate boundaries."""
        sorted_blocks = {key: [] for key in boundaries.keys()}
        for word in self.words:
            word_x = word.get_centre_coord()[0]
            for key, (x_min, x_max) in boundaries.items():
                if x_min <= word_x <= x_max:
                    sorted_blocks[key].append(word)
                    break
        return sorted_blocks

class OCRResultLine(BaseModel):
    """The complete result for a single image."""
    total_lines: int
    lines: List[TextLine]

class OCRResult(BaseModel):
    """The complete result for a single image."""
    total_blocks: int
    blocks: List[TextBlock]
    

    def order_block_left_to_right(self):
        """Sorts the blocks in place from left to right based on their x-coordinates."""
        self.blocks.sort(key=lambda block: block.get_centre_coord()[0])

    def contains_all(self, search_strings: List[str], case_sensitive: bool = False) -> bool:
        """
        Returns True only if every string in search_strings is found 
        at least once within the OCR text blocks.
        """
        # Join all text into one searchable string for efficiency
        full_text = " ".join([block.text for block in self.blocks])
        
        if not case_sensitive:
            full_text = full_text.lower()

        for search_str in search_strings:
            target = search_str if case_sensitive else search_str.lower()
            if target not in full_text:
                return False
                
        return True
    
    def get_string_center(self, target_text: str, case_sensitive: bool = False) -> Optional[Tuple[int, int]]:
        """
        Searches for an exact match of target_text within the blocks.
        Returns the center (x, y) of the first block that matches.
        """
        search_val = target_text if case_sensitive else target_text.lower()

        for block in self.blocks:
            current_text = block.text if case_sensitive else block.text.lower()
            
            # Check for exact match
            if search_val == current_text:
                return block.get_centre_coord()
                
        return None

    def find_block_by_string(self, target_text: str, case_sensitive: bool = False) -> Optional[TextBlock]:
        """
        Searches for an exact match of target_text within the blocks.
        Returns the first TextBlock that matches, or None if not found.
        """
        search_val = target_text if case_sensitive else target_text.lower()

        for block in self.blocks:
            current_text = block.text if case_sensitive else block.text.lower()
            
            # Check for exact match
            if search_val == current_text:
                return block
                
        return None


    def find_blocks_by_regex(self, pattern: str) -> List[TextBlock]:
        """
        Searches all blocks for text matching the provided regex pattern.
        Returns a list of TextBlock objects that contain at least one match.
        """
        matched_blocks = []
        # Compile the regex for better performance during the loop
        regex = re.compile(pattern)

        for block in self.blocks:
            if regex.search(block.text.replace(" ", "")):
                matched_blocks.append(block)
                
        return matched_blocks


    def sort_into_lines(self, y_tolerance_ratio=0.5) -> OCRResultLine:
        """
        Sorts individual text blocks into lines using y and x midpoints.
        """
        if not self.blocks:
            return []

        # 1. Calculate midpoints and heights for all blocks
        enriched_blocks = []
        for b in self.blocks:
            xs = [pt[0] for pt in b.box]
            ys = [pt[1] for pt in b.box]
            mid_x = sum(xs) / 4.0
            mid_y = sum(ys) / 4.0
            height = max(ys) - min(ys)
            enriched_blocks.append({"block": b, "mid_x": mid_x, "mid_y": mid_y, "height": height})

        # 2. Sort all blocks from top to bottom based on y-midpoint
        enriched_blocks.sort(key=lambda item: item["mid_y"])

        lines = []
        current_line = [enriched_blocks[0]]
        
        # 3. Group into lines
        for item in enriched_blocks[1:]:
            # Use the first word in the current line as the reference point
            line_ref_y = current_line[0]["mid_y"]
            line_ref_height = current_line[0]["height"]
            
            # If the y-midpoint is within tolerance (e.g., 50% of the reference box height), it's the same line
            if abs(item["mid_y"] - line_ref_y) < (line_ref_height * y_tolerance_ratio):
                current_line.append(item)
            else:
                # We hit a new line. Sort the completed line left-to-right via x-midpoint
                current_line.sort(key=lambda x: x["mid_x"])
                lines.append(current_line)
                # Start a new line
                current_line = [item]

        # Don't forget to add and sort the final line
        if current_line:
            current_line.sort(key=lambda x: x["mid_x"])
            lines.append(current_line)

        # 4. Convert back to Pydantic objects
        text_lines = []
        for line_items in lines:
            sorted_blocks = [item["block"] for item in line_items]
            # Create a full sentence string for the line
            combined_text = " ".join([b.text for b in sorted_blocks])
            text_lines.append(TextLine(line_text=combined_text, words=sorted_blocks))

        return OCRResultLine(total_lines=len(text_lines), lines=text_lines)


class PPOCRv5OpenVINO:
    def __init__(self, det_model_paths: List[Union[str, Path]], rec_model_paths: List[Union[str, Path]], dict_path: str):
        self.core = ov.Core()
        
        self.det_models: List[Tuple[ov.CompiledModel, ov.Input, ov.Output]] = []
        for det_model_path in det_model_paths:
            # Load Detection
            det_compiled = self.core.compile_model(det_model_path, "CPU")
            det_input = det_compiled.input(0)
            det_output = det_compiled.output(0)
            self.det_models.append((det_compiled, det_input, det_output))
        
        self.rec_models: List[Tuple[ov.CompiledModel, ov.Input, ov.Output]] = []
        for rec_model_path in rec_model_paths:
            # Load Recognition (Dynamic Width)
            rec_model = self.core.read_model(rec_model_path)
            for input_layer in rec_model.inputs:
                input_shape = input_layer.partial_shape
                input_shape[3] = -1 
                rec_model.reshape({input_layer: input_shape})
            
            rec_compiled = self.core.compile_model(rec_model, "CPU")
            rec_input = rec_compiled.input(0)
            rec_output = rec_compiled.output(0)

            self.rec_models.append((rec_compiled, rec_input, rec_output))
        
        # assume all rec models use the same dict
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.char_dict = ['blank'] + [line.strip("\n") for line in f.readlines()] + [' ']

    def preprocess_det(self, img):
        h, w = img.shape[:2]
        new_h = int(math.ceil(h / 32) * 32)
        new_w = int(math.ceil(w / 32) * 32)
        resized_img = cv2.resize(img, (new_w, new_h))
        
        resized_img = resized_img.astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        resized_img = (resized_img - mean) / std
        resized_img = resized_img.transpose((2, 0, 1))
        return np.expand_dims(resized_img, axis=0), ((h, w), (new_h, new_w))

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_rotate_crop_image(self, img, points):
        points = self.order_points_clockwise(points)
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        
        if dst_img.shape[0] * 1.0 / dst_img.shape[1] >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def preprocess_rec(self, img_crop):
        h, w = img_crop.shape[:2]
        ratio = w / float(h)
        new_h = 48 # Strict PP-OCRv5 requirement
        new_w = max(int(new_h * ratio), 48)
        
        resized = cv2.resize(img_crop, (new_w, new_h))
        resized = resized.astype('float32') / 255.0
        resized = (resized - 0.5) / 0.5
        resized = resized.transpose((2, 0, 1))
        return np.expand_dims(resized, axis=0)

    def postprocess_det(self, pred, shape_info, thresh=0.3, det_box_thresh=0.6, unclip_ratio=1.5):
        """Returns both the boxes AND the detection scores."""
        orig_h, orig_w = shape_info[0]
        new_h, new_w = shape_info[1]
        
        pred_map = pred[0, 0, :, :]
        segmentation = pred_map > thresh
        
        boxes = []
        scores = []
        contours, _ = cv2.findContours((segmentation * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4: continue
            
            poly = Polygon(points)
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            if len(expanded) == 0: continue
            
            expanded_points = np.array(expanded[0])
            rect = cv2.minAreaRect(expanded_points)
            box = cv2.boxPoints(rect)
            
            box[:, 0] = np.clip(box[:, 0] * (orig_w / new_w), 0, orig_w)
            box[:, 1] = np.clip(box[:, 1] * (orig_h / new_h), 0, orig_h)
            
            # Calculate detection score
            mask = np.zeros(pred_map.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            score = cv2.mean(pred_map, mask=mask)[0]
            
            # Filter by detection threshold early
            if score >= det_box_thresh:
                boxes.append(box)
                scores.append(score)
                
        return boxes, scores

    def postprocess_rec(self, pred):
        pred = pred[0]
        preds_idx = pred.argmax(axis=1)
        preds_prob = pred.max(axis=1)
        
        text = ""
        probs = []
        for i in range(len(preds_idx)):
            if preds_idx[i] != 0 and not (i > 0 and preds_idx[i - 1] == preds_idx[i]):
                idx = preds_idx[i]
                if idx < len(self.char_dict):
                    text += self.char_dict[idx]
                    probs.append(preds_prob[i])
        
        score = sum(probs) / len(probs) if probs else 0.0
        return text, score

    def run_recognition(self, image: np.ndarray, model_idx: int = 0) -> str:
        """Run recognition model on image"""
        try:
            rec_model, rec_input, rec_output = self.rec_models[model_idx]
            rec_input_tensor = self.preprocess_rec(image)
            rec_out = rec_model([rec_input_tensor])[rec_output]
            text, rec_score = self.postprocess_rec(rec_out)
            return text
        except Exception as e:
            logger.error(f"Error in recognition model: {str(e)}")
    
    def run_detection(self, img: Optional[np.ndarray], det_thresh=0.8, visualise=False, model_idx: int = 0) -> OCRResult:
        """Runs pipeline and filters by both det and rec thresholds."""
        if img is None:
            raise ValueError(f"Img is None")
        
        det_compiled, det_input, det_output = self.det_models[model_idx]

        # 1. Detection
        det_input_tensor, shape_info = self.preprocess_det(img)
        det_out = det_compiled([det_input_tensor])[det_output]
        boxes, det_scores = self.postprocess_det(det_out, shape_info, det_box_thresh=det_thresh)
        
        text_blocks = []
        
        # 2. Recognition & Filtering
        for box, det_score in zip(boxes, det_scores):
            clean_box = [(float(pt[0]), float(pt[1])) for pt in box.tolist()]
            
            text_blocks.append(TextBlock(
                box=clean_box,
                text="",
                det_score=float(det_score),
                rec_score=1.0
            ))
    
        # 3. Assemble Pydantic Object
        final_result = OCRResult(
            total_blocks=len(text_blocks),
            blocks=text_blocks
        )
        if visualise:
            self.visualize(img, final_result)
        
        return final_result     

    def run(self, img: Optional[np.ndarray], det_thresh=0.85, rec_thresh=0.8, visualise=False, det_model_idx: int =0, rec_model_idx: int =0) -> OCRResult:
        """Runs pipeline and filters by both det and rec thresholds."""
        if img is None:
            raise ValueError(f"Img is None")
        
        det_compiled, det_input, det_output = self.det_models[det_model_idx]
        rec_compiled, rec_input, rec_output = self.rec_models[rec_model_idx]


        # 1. Detection
        det_input_tensor, shape_info = self.preprocess_det(img)
        det_out = det_compiled([det_input_tensor])[det_output]
        boxes, det_scores = self.postprocess_det(det_out, shape_info, det_box_thresh=det_thresh)
        
        text_blocks = []
        
        # 2. Recognition & Filtering
        for box, det_score in zip(boxes, det_scores):
            crop = self.get_rotate_crop_image(img, box)
            
            # Skip if crop is invalid (e.g., box too small)
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            rec_input_tensor = self.preprocess_rec(crop)
            rec_out = rec_compiled([rec_input_tensor])[rec_output]
            text, rec_score = self.postprocess_rec(rec_out)
            # Filter by Recognition threshold
            if rec_score >= rec_thresh:
                # Store coordinates as simple Python floats for Pydantic
                clean_box = [(float(pt[0]), float(pt[1])) for pt in box.tolist()]
                
                text_blocks.append(TextBlock(
                    box=clean_box,
                    text=text,
                    det_score=float(det_score),
                    rec_score=float(rec_score)
                ))
        
        # 3. Assemble Pydantic Object
        final_result = OCRResult(
            total_blocks=len(text_blocks),
            blocks=text_blocks
        )
        if visualise:
            self.visualize(img, final_result)
        
        return final_result
    

    def visualize(self, img: np.ndarray, ocr_result: OCRResult, save_path="ppocrv5_filtered_result.jpg"):
        """Visualizes the final Pydantic object."""
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        for block in ocr_result.blocks:
            poly = [(pt[0], pt[1]) for pt in block.box]
            draw.polygon(poly, outline="red", width=2)
            
            text_str = f"{block.text} ({block.rec_score:.2f})"
            bbox = font.getbbox(text_str)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            top_left = (block.box[0][0], block.box[0][1] - text_h - 4)
            draw.rectangle([top_left, (top_left[0] + text_w + 4, top_left[1] + text_h + 4)], fill="red")
            draw.text((top_left[0] + 2, top_left[1] + 2), text_str, fill="white", font=font)
            
        image.save(save_path)
        print(f"Visualization saved to {save_path}")

def extract_numeric_value(text: str, money: bool = False) -> Optional[float]:
    """Extract numeric value and handle K/M conversion"""
    try:
        # Remove all non-numeric characters except K, M, and decimal point
        cleaned = ''.join(c for c in text if c.isdigit() or c in 'KM.')
        
        # Find number and unit
        match = re.match(r'(\d+\.?\d*)([KM])?', cleaned)
        if not match:
            return None
            
        number = float(match.group(1))
        unit = match.group(2)
        
        if money:
            # Convert K to M if necessary
            if unit == 'K':
                return number / 1000
            if unit == None:
                return number / 1000000
        return number
    except Exception as e:
        logger.error(f"Error extracting numeric value from text '{text}': {str(e)}")
        return None

def get_player_name(name_text: str) -> Optional[str]:
    """Extract player name from the specified region"""
    try:
        logger.info(f"Found name: {name_text}")
        last_dot_index = name_text.rfind('.')
        last_comma_index = name_text.rfind(',')
        
        # Find the rightmost dot/comma
        if last_dot_index > last_comma_index:
            split_index = last_dot_index
        else:
            split_index = last_comma_index
            
        # If no dot/comma found, or it is the first character, return original
        if split_index <= 0:
            return name_text.strip()
            
        # Get one char before and everything after the dot/comma
        output = name_text[split_index-1:].strip()
        logger.info(f"Extracted name: {output}")
        return output
        
    except Exception as e:
        logger.error(f"Error processing player name: {str(e)}")
        return None