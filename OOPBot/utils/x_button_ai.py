import cv2
import numpy as np
import torch
from pathlib import Path
import onnxruntime
from typing import List, Optional, Union
from torchvision.ops import nms
from OOPBot.interface import ScreenRegion

MODEL_PATH = Path("x_detection_model/x_detector.onnx")
INPUT_SIZE = (960, 960)
CONF_THRESHOLD = 0.7
NMS_IOU = 0.5

_onnx_session = None

def get_onnx_session(model_path: Union[str, Path] = MODEL_PATH):
    global _onnx_session
    if _onnx_session is None:
        _onnx_session = onnxruntime.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    return _onnx_session

def preprocess_image(image: np.ndarray, input_size=INPUT_SIZE):
    height, width = image.shape[:2]
    target_w, target_h = input_size
    scale = min(target_w / width, target_h / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    normalized_image = rgb_image.astype(np.float32) / 255.0
    image_np = np.transpose(normalized_image, (2, 0, 1))[np.newaxis, ...]
    scale_info = (scale, left, top)
    return image_np, scale_info

def nms_detections(detections, iou_threshold=NMS_IOU):
    if not detections or len(detections) == 0:
        return []
    dets = np.array(detections)
    boxes = torch.tensor(dets[:, :4], dtype=torch.float32)
    scores = torch.tensor(dets[:, 4], dtype=torch.float32)
    class_ids = dets[:, 5]
    keep = []
    for cls in np.unique(class_ids):
        idxs = np.where(class_ids == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]
        keep_idxs = nms(cls_boxes, cls_scores, iou_threshold)
        keep.extend(idxs[np.atleast_1d(keep_idxs)].tolist())
    return [detections[i] for i in keep]

def detect_x_buttons(
    image: Union[str, np.ndarray],
    model_path: Union[str, Path] = MODEL_PATH,
    input_size=INPUT_SIZE,
    conf_threshold=CONF_THRESHOLD,
    nms_iou=NMS_IOU
) -> List[ScreenRegion]:
    """
    Detect X buttons in an image using the ONNX model. Returns a list of ScreenRegion objects.
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return []
    image_np, scale_info = preprocess_image(image, input_size)
    scale, pad_x, pad_y = scale_info
    session = get_onnx_session(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_np})
    output = outputs[0]
    if len(output.shape) == 3:
        output = output[0]
    detections = []
    for det in output:
        if det.shape[-1] == 6:
            x_c, y_c, w, h, conf, cls = det.tolist()
        else:
            x_c, y_c, w, h, obj_conf, *cls_confs = det.tolist()
            conf = obj_conf * max(cls_confs)
            cls = np.argmax(cls_confs)
        if conf >= conf_threshold:
            x_c_orig = (x_c - pad_x) / scale
            y_c_orig = (y_c - pad_y) / scale
            w_orig = w / scale
            h_orig = h / scale
            x1 = x_c_orig - w_orig / 2
            y1 = y_c_orig - h_orig / 2
            x2 = x_c_orig + w_orig / 2
            y2 = y_c_orig + h_orig / 2
            detections.append([x1, y1, x2, y2, conf, cls])
    detections = nms_detections(detections, iou_threshold=nms_iou)
    regions = [ScreenRegion(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)) for x1, y1, x2, y2, conf, cls in detections]
    return regions 