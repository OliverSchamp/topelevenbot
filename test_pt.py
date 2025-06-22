import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import onnxruntime
import time
from torchvision.ops import nms

def crop_black_bars(image, black_thresh=10, min_bar_thickness_ratio=0.05):
    """
    Detect and crop large black bars (letterbox) from the image.
    Args:
        image: Input BGR image (numpy array)
        black_thresh: Pixel value threshold to consider as black (0-255)
        min_bar_thickness_ratio: Minimum thickness ratio (w.r.t. image size) to consider as a bar
    Returns:
        Cropped image (if bars found), else original image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold for very dark pixels
    _, mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    h, w = mask.shape
    # Sum mask along axes to find black regions
    col_sum = np.sum(mask, axis=0) / 255  # sum of black pixels per column
    row_sum = np.sum(mask, axis=1) / 255  # sum of black pixels per row
    # Thresholds for what counts as a 'fully black' bar
    col_black_thresh = h * 0.98  # 98% of pixels in column are black
    row_black_thresh = w * 0.98  # 98% of pixels in row are black
    # Find left bar
    left = 0
    while left < w and col_sum[left] > col_black_thresh:
        left += 1
    # Find right bar
    right = w - 1
    while right >= 0 and col_sum[right] > col_black_thresh:
        right -= 1
    # Find top bar
    top = 0
    while top < h and row_sum[top] > row_black_thresh:
        top += 1
    # Find bottom bar
    bottom = h - 1
    while bottom >= 0 and row_sum[bottom] > row_black_thresh:
        bottom -= 1
    # Only crop if bars are significant
    min_bar_w = int(w * min_bar_thickness_ratio)
    min_bar_h = int(h * min_bar_thickness_ratio)
    crop_left = left if left > min_bar_w else 0
    crop_right = right if (w - 1 - right) > min_bar_w else w - 1
    crop_top = top if top > min_bar_h else 0
    crop_bottom = bottom if (h - 1 - bottom) > min_bar_h else h - 1
    # If any cropping is to be done
    if crop_left > 0 or crop_right < w - 1 or crop_top > 0 or crop_bottom < h - 1:
        cropped = image[crop_top:crop_bottom+1, crop_left:crop_right+1]
        return cropped
    return image

def preprocess_image(image_path, input_size=(960, 960)):
    """
    Preprocess image for YOLOv5 ONNX inference with letterbox resize (aspect ratio preserved, zero padding).
    Returns numpy array for ONNX input, original image, and scale/pad info.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None, None, None
        # Crop black bars if present
        image = crop_black_bars(image)
        original_image = image.copy()
        height, width = image.shape[:2]
        target_w, target_h = input_size
        # Compute scale and padding
        scale = min(target_w / width, target_h / height)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        # #display the image
        # cv2.imshow("padded_image", padded_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # padded_image_2, _, _ = letterbox(image, (960, 960))
        # cv2.imshow("padded_image_2", padded_image_2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        normalized_image = rgb_image.astype(np.float32) / 255.0
        image_np = np.transpose(normalized_image, (2, 0, 1))[np.newaxis, ...]
        # For reverse mapping: scale, pad (left, top)
        scale_info = (scale, left, top)
        return image_np, original_image, scale_info
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None, None

def nms_detections(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to a list of detections.
    Args:
        detections: List of [x1, y1, x2, y2, conf, class_id]
        iou_threshold: IoU threshold for NMS
    Returns:
        List of filtered detections after NMS
    """
    if not detections or len(detections) == 0:
        return []
    dets = np.array(detections)
    boxes = torch.tensor(dets[:, :4], dtype=torch.float32)
    scores = torch.tensor(dets[:, 4], dtype=torch.float32)
    # Optionally, you can do class-wise NMS
    class_ids = dets[:, 5]
    keep = []
    for cls in np.unique(class_ids):
        idxs = np.where(class_ids == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]
        keep_idxs = nms(cls_boxes, cls_scores, iou_threshold)
        keep.extend(idxs[np.atleast_1d(keep_idxs)].tolist())
    return [detections[i] for i in keep]

def run_inference_onnx(session, image_path, conf_threshold=0.02, input_size=(960, 960), nms_iou=0.5):
    """
    Run inference on an image using the loaded ONNX model session.
    Args:
        session: ONNXRuntime InferenceSession
        image_path: Path to the input image
        conf_threshold: Confidence threshold for detections
        input_size: Input size for the model
        nms_iou: IoU threshold for NMS
    Returns:
        detections: List of detections with format [x1, y1, x2, y2, confidence, class_id]
    """
    image_np, original_image, scale_info = preprocess_image(image_path, input_size)
    if image_np is None:
        return None
    scale, pad_x, pad_y = scale_info
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_np})
    output = outputs[0]
    if len(output.shape) == 3:
        output = output[0]  # Remove batch dim
    detections = []
    for det in output:
        if det.shape[-1] == 6:
            x_c, y_c, w, h, conf, cls = det.tolist()
        else:
            x_c, y_c, w, h, obj_conf, *cls_confs = det.tolist()
            conf = obj_conf * max(cls_confs)
            cls = np.argmax(cls_confs)
        if conf >= conf_threshold:
            # Map from letterboxed image to original image
            x_c_orig = (x_c - pad_x) / scale
            y_c_orig = (y_c - pad_y) / scale
            w_orig = w / scale
            h_orig = h / scale
            x1 = x_c_orig - w_orig / 2
            y1 = y_c_orig - h_orig / 2
            x2 = x_c_orig + w_orig / 2
            y2 = y_c_orig + h_orig / 2
            detections.append([x1, y1, x2, y2, conf, cls])
    # Apply NMS
    detections = nms_detections(detections, iou_threshold=nms_iou)
    return detections


def visualize_results(image_path, detections, output_path=None):
    """
    Visualize detection results on the image
    Args:
        image_path: Path to the input image
        detections: List of detections from run_inference
        output_path: Path to save the output image (optional)
    """
    try:
        image = cv2.imread(image_path)
        image = crop_black_bars(image)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        class_names = ['class_0', 'class_1']  # Update with your actual class names
        if detections and len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2, conf, class_id = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_id = int(class_id)
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Detection: {class_name} at ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.2f}")
        else:
            print("No detections found")
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Result saved to: {output_path}")
        else:
            cv2.imshow('Detection Results', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error visualizing results: {e}")

def main():
    """
    Main function to demonstrate ONNX model loading and inference
    """
    model_path = "x_detection_model/x_detector.onnx"
    if not os.path.exists(model_path):
        print(f"ONNX model not found at: {model_path}")
        return
    print("Loading ONNX model...")
    try:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return
    print("\nONNX model loaded successfully!")
    print("To run inference on an image, use:")
    print("detections = run_inference_onnx(session, 'path_to_your_image.jpg')")
    print("visualize_results('path_to_your_image.jpg', detections)")
    test_image_path = "img/auto_auction/new_ads_img/5.jpg"
    if os.path.exists(test_image_path):
        print(f"\nRunning inference on {test_image_path}...")
        start_time = time.time()
        detections = run_inference_onnx(session, test_image_path)
        end_time = time.time()
        print(f"Inference time: {(end_time - start_time)*1000} ms")
        visualize_results(test_image_path, detections, "output_result.jpg")

if __name__ == "__main__":
    main() 