import cv2
import numpy as np
import openvino as ov
import pyclipper
import math
from pydantic import BaseModel, Field
from typing import List, Tuple
from shapely.geometry import Polygon
from PIL import Image, ImageDraw, ImageFont

# --- 1. Updated Pydantic Models ---

class TextBlock(BaseModel):
    """Represents a single detected word/phrase."""
    box: List[Tuple[float, float]] = Field(description="List of 4 (x, y) coordinates: [top-left, top-right, bottom-right, bottom-left]")
    text: str = Field(description="The recognized text string")
    det_score: float = Field(description="Confidence score of the text detection")
    rec_score: float = Field(description="Confidence score of the text recognition")

class TextLine(BaseModel):
    """Represents a full horizontal line of text."""
    line_text: str = Field(description="The combined text of all words in this line")
    words: List[TextBlock] = Field(description="The individual text blocks, ordered left-to-right")

class OCRResult(BaseModel):
    """The complete result for a single image."""
    image_path: str
    total_lines: int
    lines: List[TextLine]


# --- 2. The OpenVINO Pipeline ---

class PPOCRv5OpenVINO:
    def __init__(self, det_model_path, rec_model_path, dict_path):
        self.core = ov.Core()
        
        # Load Detection
        self.det_compiled = self.core.compile_model(det_model_path, "CPU")
        self.det_input = self.det_compiled.input(0)
        self.det_output = self.det_compiled.output(0)
        
        # Load Recognition
        rec_model = self.core.read_model(rec_model_path)
        for input_layer in rec_model.inputs:
            input_shape = input_layer.partial_shape
            input_shape[3] = -1 
            rec_model.reshape({input_layer: input_shape})
        
        self.rec_compiled = self.core.compile_model(rec_model, "CPU")
        self.rec_input = self.rec_compiled.input(0)
        self.rec_output = self.rec_compiled.output(0)
        
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.char_dict = ['blank'] + [line.strip("\n") for line in f.readlines()] + [' ']

    # ... [Keep preprocess_det, order_points_clockwise, get_rotate_crop_image, preprocess_rec, postprocess_det, postprocess_rec the same as before] ...

    def preprocess_det(self, img):
        h, w = img.shape[:2]
        new_h = int(math.ceil(h / 32) * 32) // 2
        new_w = int(math.ceil(w / 32) * 32) // 2
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


    # -> Methods omitted for brevity, insert the previous processing functions here <-

    def sort_into_lines(self, blocks: List[TextBlock], y_tolerance_ratio=0.5) -> List[TextLine]:
        """
        Sorts individual text blocks into lines using y and x midpoints.
        """
        if not blocks:
            return []

        # 1. Calculate midpoints and heights for all blocks
        enriched_blocks = []
        for b in blocks:
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

        return text_lines

    def visualize(self, image_path, ocr_result: OCRResult, save_path="ppocrv5_line_result.jpg"):
        """Visualizes the organized line output."""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        # We can iterate through lines and words to draw them
        for line in ocr_result.lines:
            for block in line.words:
                poly = [(pt[0], pt[1]) for pt in block.box]
                draw.polygon(poly, outline="red", width=2)
                
                text_str = f"{block.text}"
                bbox = font.getbbox(text_str)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                top_left = (block.box[0][0], block.box[0][1] - text_h - 4)
                draw.rectangle([top_left, (top_left[0] + text_w + 4, top_left[1] + text_h + 4)], fill="red")
                draw.text((top_left[0] + 2, top_left[1] + 2), text_str, fill="white", font=font)
            
        image.save(save_path)
        print(f"Visualization saved to {save_path}")

    def run(self, img_path, det_thresh=0.6, rec_thresh=0.5) -> OCRResult:
        """Runs pipeline, filters, and groups into lines."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
            
        # 1. Detection
        det_input_tensor, shape_info = self.preprocess_det(img)
        det_out = self.det_compiled([det_input_tensor])[self.det_output]
        boxes, det_scores = self.postprocess_det(det_out, shape_info, det_box_thresh=det_thresh)
        
        raw_blocks = []
        
        # 2. Recognition
        for box, det_score in zip(boxes, det_scores):
            crop = self.get_rotate_crop_image(img, box)
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            rec_input_tensor = self.preprocess_rec(crop)
            rec_out = self.rec_compiled([rec_input_tensor])[self.rec_output]
            text, rec_score = self.postprocess_rec(rec_out)
            
            if rec_score >= rec_thresh:
                clean_box = [(float(pt[0]), float(pt[1])) for pt in box.tolist()]
                raw_blocks.append(TextBlock(
                    box=clean_box,
                    text=text,
                    det_score=float(det_score),
                    rec_score=float(rec_score)
                ))
        
        # 3. Sort into lines based on midpoints
        text_lines = self.sort_into_lines(raw_blocks, y_tolerance_ratio=0.5)
        
        # 4. Assemble final Pydantic Object
        final_result = OCRResult(
            image_path=img_path,
            total_lines=len(text_lines),
            lines=text_lines
        )
        
        self.visualize(img_path, final_result)
        return final_result

# --- Execution ---
if __name__ == "__main__":
    # Note: Copy in the preprocess and postprocess methods from the previous block.
    ocr_pipeline = PPOCRv5OpenVINO("ocr_model/detector/detector.xml", "ocr_model/recognizer/recognizer.xml", "ocr_model/ppocrv5_en_dict.txt")
    result_obj = ocr_pipeline.run("templates/test4.png", det_thresh=0.85, rec_thresh=0.85)
    print(result_obj.model_dump_json(indent=2))
    with open("outlines.json", "w") as f:
        f.write(result_obj.model_dump_json(indent=2))