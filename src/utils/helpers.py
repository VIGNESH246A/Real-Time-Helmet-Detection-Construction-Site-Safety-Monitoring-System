"""
Helper utility functions
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Create formatted timestamp string
    
    Args:
        format_str: Datetime format string
        
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format_str)


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Place resized image on canvas
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return canvas


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    padding: int = 5
) -> np.ndarray:
    """
    Draw text with background rectangle
    
    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position
        font_scale: Font scale
        color: Text color (BGR)
        bg_color: Background color (BGR)
        thickness: Text thickness
        padding: Background padding
        
    Returns:
        Image with text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return image


def calculate_fps(prev_time: float, curr_time: float) -> float:
    """
    Calculate frames per second
    
    Args:
        prev_time: Previous timestamp
        curr_time: Current timestamp
        
    Returns:
        FPS value
    """
    time_diff = curr_time - prev_time
    return 1.0 / time_diff if time_diff > 0 else 0


def clip_boxes_to_image(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries
    
    Args:
        boxes: Array of boxes [x1, y1, x2, y2]
        image_shape: (height, width)
        
    Returns:
        Clipped boxes
    """
    h, w = image_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
    
    Args:
        boxes: Boxes in xywh format
        
    Returns:
        Boxes in xyxy format
    """
    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
    return boxes_xyxy


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [x1, y1, x2, y2] to [x, y, w, h]
    
    Args:
        boxes: Boxes in xyxy format
        
    Returns:
        Boxes in xywh format
    """
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
    return boxes_xywh


def get_box_center(box: np.ndarray) -> Tuple[int, int]:
    """
    Get center point of bounding box
    
    Args:
        box: [x1, y1, x2, y2]
        
    Returns:
        (center_x, center_y)
    """
    cx = int((box[0] + box[2]) / 2)
    cy = int((box[1] + box[3]) / 2)
    return cx, cy


def get_box_area(box: np.ndarray) -> float:
    """
    Calculate bounding box area
    
    Args:
        box: [x1, y1, x2, y2]
        
    Returns:
        Box area
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def sanitize_filename(filename: str) -> str:
    """
    Remove invalid characters from filename
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename