import cv2
import numpy as np
import openvino as ov
import pyclipper
import math
from pydantic import BaseModel, Field
from typing import List, Tuple
from shapely.geometry import Polygon
from PIL import Image, ImageDraw, ImageFont
import time
# --- 1. Pydantic Models for Human-Readable Output ---

# class TextBlock(BaseModel):
#     """Represents a single detected and recognized line of text."""
#     box: List[Tuple[float, float]] = Field(description="List of 4 (x, y) coordinates: [top-left, top-right, bottom-right, bottom-left]")
#     text: str = Field(description="The recognized text string")
#     det_score: float = Field(description="Confidence score of the text detection bounding box")
#     rec_score: float = Field(description="Confidence score of the text recognition")

# class OCRResult(BaseModel):
#     """The complete result for a single image."""
#     image_path: str
#     total_blocks: int
#     blocks: List[TextBlock]


from OOPBot.utils.ocr import PPOCRv5OpenVINO, OCRResult, TextBlock, OCRResultLine, TextLine
from PIL import Image

# --- 2. The OpenVINO Pipeline ---

# class PPOCRv5OpenVINO:
#     def __init__(self, det_model_path, rec_model_path, dict_path):
#         self.core = ov.Core()
        
#         # Load Detection
#         self.det_compiled = self.core.compile_model(det_model_path, "CPU")
#         self.det_input = self.det_compiled.input(0)
#         self.det_output = self.det_compiled.output(0)
        
#         # Load Recognition (Dynamic Width)
#         rec_model = self.core.read_model(rec_model_path)
#         for input_layer in rec_model.inputs:
#             input_shape = input_layer.partial_shape
#             input_shape[3] = -1 
#             rec_model.reshape({input_layer: input_shape})
        
#         self.rec_compiled = self.core.compile_model(rec_model, "CPU")
#         self.rec_input = self.rec_compiled.input(0)
#         self.rec_output = self.rec_compiled.output(0)
        
#         with open(dict_path, 'r', encoding='utf-8') as f:
#             self.char_dict = ['blank'] + [line.strip("\n") for line in f.readlines()] + [' ']

#     # ... [Keep preprocess_det, order_points_clockwise, get_rotate_crop_image, preprocess_rec as in the previous script] ...
    
#     def preprocess_det(self, img):
#         h, w = img.shape[:2]
#         new_h = int(math.ceil(h / 64) * 64) // 2
#         new_w = int(math.ceil(w / 64) * 64) // 2
#         resized_img = cv2.resize(img, (new_w, new_h))
        
#         resized_img = resized_img.astype('float32') / 255.0
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         resized_img = (resized_img - mean) / std
#         resized_img = resized_img.transpose((2, 0, 1))
#         return np.expand_dims(resized_img, axis=0), ((h, w), (new_h, new_w))

#     def order_points_clockwise(self, pts):
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
#         return rect

#     def get_rotate_crop_image(self, img, points):
#         points = self.order_points_clockwise(points)
#         img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
#         img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        
#         pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
#         M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
#         dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        
#         if dst_img.shape[0] * 1.0 / dst_img.shape[1] >= 1.5:
#             dst_img = np.rot90(dst_img)
#         return dst_img

#     def preprocess_rec(self, img_crop):
#         h, w = img_crop.shape[:2]
#         ratio = w / float(h)
#         new_h = 48 # Strict PP-OCRv5 requirement
#         new_w = max(int(new_h * ratio), 48)
        
#         resized = cv2.resize(img_crop, (new_w, new_h))
#         resized = resized.astype('float32') / 255.0
#         resized = (resized - 0.5) / 0.5
#         resized = resized.transpose((2, 0, 1))
#         return np.expand_dims(resized, axis=0)

#     def postprocess_det(self, pred, shape_info, thresh=0.3, det_box_thresh=0.6, unclip_ratio=1.5):
#         """Returns both the boxes AND the detection scores."""
#         orig_h, orig_w = shape_info[0]
#         new_h, new_w = shape_info[1]
        
#         pred_map = pred[0, 0, :, :]
#         segmentation = pred_map > thresh
        
#         boxes = []
#         scores = []
#         contours, _ = cv2.findContours((segmentation * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#         for contour in contours:
#             epsilon = 0.001 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)
#             points = approx.reshape((-1, 2))
#             if points.shape[0] < 4: continue
            
#             poly = Polygon(points)
#             distance = poly.area * unclip_ratio / poly.length
#             offset = pyclipper.PyclipperOffset()
#             offset.AddPath(points.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
#             expanded = offset.Execute(distance)
#             if len(expanded) == 0: continue
            
#             expanded_points = np.array(expanded[0])
#             rect = cv2.minAreaRect(expanded_points)
#             box = cv2.boxPoints(rect)
            
#             box[:, 0] = np.clip(box[:, 0] * (orig_w / new_w), 0, orig_w)
#             box[:, 1] = np.clip(box[:, 1] * (orig_h / new_h), 0, orig_h)
            
#             # Calculate detection score
#             mask = np.zeros(pred_map.shape, dtype=np.uint8)
#             cv2.fillPoly(mask, [points.astype(np.int32)], 1)
#             score = cv2.mean(pred_map, mask=mask)[0]
            
#             # Filter by detection threshold early
#             if score >= det_box_thresh:
#                 boxes.append(box)
#                 scores.append(score)
                
#         return boxes, scores

#     def postprocess_rec(self, pred):
#         pred = pred[0]
#         preds_idx = pred.argmax(axis=1)
#         preds_prob = pred.max(axis=1)
        
#         text = ""
#         probs = []
#         for i in range(len(preds_idx)):
#             if preds_idx[i] != 0 and not (i > 0 and preds_idx[i - 1] == preds_idx[i]):
#                 idx = preds_idx[i]
#                 if idx < len(self.char_dict):
#                     text += self.char_dict[idx]
#                     probs.append(preds_prob[i])
        
#         score = sum(probs) / len(probs) if probs else 0.0
#         return text, score

#     def visualize(self, image_path, ocr_result: OCRResult, save_path="ppocrv5_filtered_result.jpg"):
#         """Visualizes the final Pydantic object."""
#         image = Image.open(image_path).convert("RGB")
#         draw = ImageDraw.Draw(image)
        
#         try:
#             font = ImageFont.truetype("arial.ttf", 20)
#         except IOError:
#             font = ImageFont.load_default()
            
#         for block in ocr_result.blocks:
#             poly = [(pt[0], pt[1]) for pt in block.box]
#             draw.polygon(poly, outline="red", width=2)
            
#             text_str = f"{block.text} ({block.rec_score:.2f})"
#             bbox = font.getbbox(text_str)
#             text_w = bbox[2] - bbox[0]
#             text_h = bbox[3] - bbox[1]
            
#             top_left = (block.box[0][0], block.box[0][1] - text_h - 4)
#             draw.rectangle([top_left, (top_left[0] + text_w + 4, top_left[1] + text_h + 4)], fill="red")
#             draw.text((top_left[0] + 2, top_left[1] + 2), text_str, fill="white", font=font)
            
#         image.save(save_path)
#         print(f"Visualization saved to {save_path}")

#     def run(self, img_path, det_thresh=0.6, rec_thresh=0.5) -> OCRResult:
#         """Runs pipeline and filters by both det and rec thresholds."""
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError(f"Could not read image: {img_path}")
            
#         # 1. Detection
#         print(img.shape)
#         det_input_tensor, shape_info = self.preprocess_det(img)
#         print(shape_info, det_input_tensor.shape)
#         det_out = self.det_compiled([det_input_tensor])[self.det_output]
#         boxes, det_scores = self.postprocess_det(det_out, shape_info, det_box_thresh=det_thresh)
        
#         text_blocks = []
        
#         # 2. Recognition & Filtering
#         for box, det_score in zip(boxes, det_scores):
#             crop = self.get_rotate_crop_image(img, box)
            
#             # Skip if crop is invalid (e.g., box too small)
#             if crop.shape[0] == 0 or crop.shape[1] == 0:
#                 continue
                
#             rec_input_tensor = self.preprocess_rec(crop)
#             rec_out = self.rec_compiled([rec_input_tensor])[self.rec_output]
#             text, rec_score = self.postprocess_rec(rec_out)
#             # Filter by Recognition threshold
#             if rec_score >= rec_thresh:
#                 # Store coordinates as simple Python floats for Pydantic
#                 clean_box = [(float(pt[0]), float(pt[1])) for pt in box.tolist()]
                
#                 text_blocks.append(TextBlock(
#                     box=clean_box,
#                     text=text,
#                     det_score=float(det_score),
#                     rec_score=float(rec_score)
#                 ))
        
#         # 3. Assemble Pydantic Object
#         final_result = OCRResult(
#             image_path=img_path,
#             total_blocks=len(text_blocks),
#             blocks=text_blocks
#         )
        
#         # 4. Visualize
#         self.visualize(img_path, final_result)
        
#         return final_result


#     def run_recognition(self, img_path) -> Tuple[str, float]:
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError(f"Could not read image: {img_path}")
        
#         try:
#             start = time.time()
#             rec_input_tensor = self.preprocess_rec(img)
#             rec_out = self.rec_compiled([rec_input_tensor])[self.rec_output]
#             text, rec_score = self.postprocess_rec(rec_out)
#             print(time.time() - start)
#             return text, rec_score
#         except Exception as e:
#             print(f"Error in recognition model: {str(e)}")

# --- Execution ---
if __name__ == "__main__":
    ocr_pipeline = PPOCRv5OpenVINO(
        det_model_paths=["ocr_model/detector/detector.xml", "ocr_model/detector_table/output_det_ft.xml"],
        rec_model_paths=["ocr_model/recognizer/recognizer.xml"],
        dict_path="ocr_model/ppocrv5_en_dict.txt"
    )

    img = Image.open("debug/transfers_page_20260329_200323.jpg").convert("RGB")
    img = np.array(img)

    start = time.time()
    # Run with custom thresholds
    # result_obj: OCRResult = ocr_pipeline.run(img, det_thresh=0.95, rec_thresh=0.4, det_model_idx=1, visualise=True)
    result_obj: OCRResult = ocr_pipeline.run(img, visualise=True)
    input("stop")
    result_obj_lines: OCRResultLine = result_obj.sort_into_lines()
    for idx, line in enumerate(result_obj_lines.lines):
        line_y_min, line_y_max = line.get_line_y_min_y_max()

        line_image = img[line_y_min:line_y_max, :]

        lineresult: OCRResult = ocr_pipeline.run(line_image, det_thresh=0.1, rec_thresh=0.1, visualise=True)
        lineresult.order_block_left_to_right()

        # convert lineresult into a textline object that replaces the original line
        textline = TextLine(
            line_text=" ".join([block.text for block in lineresult.blocks]),
            # add line_y_min-5 to the block y coordinates to get the correct position in the original image
            words=[TextBlock(
                box=[(pt[0], pt[1]+line_y_min) for pt in block.box],
                text=block.text,
                det_score=block.det_score,
                rec_score=block.rec_score) for block in lineresult.blocks]
        )
        result_obj_lines.lines[idx] = textline
        input(f"Line {idx+1} Text: {textline.line_text}")
    
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    # # Easily view the human-readable structured data

    # with open("out.json", "w") as f:
    #     f.write(result_obj.model_dump_json(indent=2))
    # # print(result_obj.model_dump_json(indent=2))

    # start = time.time()
    # print(ocr_pipeline.run_recognition("/mnt/SF_NAS/Oliver/image_problem4.jpg"))
    # print(time.time() - start)