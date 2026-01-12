"""
Video frame visualization and annotation
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FrameVisualizer:
    """
    Annotate and visualize detection frames
    """
    
    def __init__(
        self,
        show_confidence: bool = True,
        show_fps: bool = True,
        show_stats: bool = True
    ):
        """
        Initialize visualizer
        
        Args:
            show_confidence: Show confidence scores
            show_fps: Show FPS counter
            show_stats: Show detection statistics
        """
        self.show_confidence = show_confidence
        self.show_fps = show_fps
        self.show_stats = show_stats
        
        # Colors (BGR)
        self.colors = {
            'helmet': (0, 255, 0),         # Green
            'no_helmet': (0, 0, 255),      # Red
            'person': (255, 0, 0),         # Blue
            'warning': (0, 165, 255),      # Orange
            'text_bg': (0, 0, 0),          # Black
            'text': (255, 255, 255),       # White
        }
        
        logger.info("Frame visualizer initialized")
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        fps: Optional[float] = None,
        stats: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Annotate frame with detections and information
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            fps: Current FPS
            stats: Detection statistics
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detection boxes
        for det in detections:
            annotated = self._draw_detection(annotated, det)
        
        # Draw info overlay
        if self.show_fps or self.show_stats:
            annotated = self._draw_info_overlay(annotated, fps, stats, len(detections))
        
        return annotated
    
    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Dict
    ) -> np.ndarray:
        """
        Draw single detection on frame
        
        Args:
            frame: Input frame
            detection: Detection dictionary
            
        Returns:
            Frame with detection drawn
        """
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        confidence = detection['confidence']
        class_name = detection['class_name'].lower()
        
        # Determine color based on class
        if 'helmet' in class_name and 'no' not in class_name:
            color = self.colors['helmet']
            label_prefix = "✓"
        elif 'no' in class_name:
            color = self.colors['no_helmet']
            label_prefix = "✗"
        else:
            color = self.colors['person']
            label_prefix = ""
        
        # Draw bounding box
        thickness = 3 if 'no' in class_name else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        if self.show_confidence:
            label = f"{label_prefix} {class_name}: {confidence:.2f}"
        else:
            label = f"{label_prefix} {class_name}"
        
        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 10, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            self.colors['text'],
            font_thickness,
            cv2.LINE_AA
        )
        
        return frame
    
    def _draw_info_overlay(
        self,
        frame: np.ndarray,
        fps: Optional[float],
        stats: Optional[Dict],
        detection_count: int
    ) -> np.ndarray:
        """
        Draw information overlay on frame
        
        Args:
            frame: Input frame
            fps: Current FPS
            stats: Statistics dictionary
            detection_count: Number of detections
            
        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        overlay_height = 100
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (400, overlay_height),
            (0, 0, 0),
            -1
        )
        
        # Blend overlay
        alpha = 0.7
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        y_offset = 25
        
        # Draw FPS
        if self.show_fps and fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, y_offset),
                font,
                font_scale,
                self.colors['text'],
                font_thickness,
                cv2.LINE_AA
            )
            y_offset += 25
        
        # Draw detection count
        count_text = f"Detections: {detection_count}"
        cv2.putText(
            frame,
            count_text,
            (10, y_offset),
            font,
            font_scale,
            self.colors['text'],
            font_thickness,
            cv2.LINE_AA
        )
        y_offset += 25
        
        # Draw compliance info if stats provided
        if self.show_stats and stats:
            helmet_count = stats.get('helmet_count', 0)
            no_helmet_count = stats.get('no_helmet_count', 0)
            compliance_rate = stats.get('compliance_rate', 0)
            
            compliance_text = f"Compliance: {compliance_rate*100:.1f}%"
            compliance_color = self.colors['helmet'] if compliance_rate > 0.85 else self.colors['no_helmet']
            
            cv2.putText(
                frame,
                compliance_text,
                (10, y_offset),
                font,
                font_scale,
                compliance_color,
                font_thickness,
                cv2.LINE_AA
            )
        
        return frame
    
    def draw_violation_alert(
        self,
        frame: np.ndarray,
        message: str = "⚠️ SAFETY VIOLATION DETECTED!"
    ) -> np.ndarray:
        """
        Draw prominent violation alert banner
        
        Args:
            frame: Input frame
            message: Alert message
            
        Returns:
            Frame with alert
        """
        h, w = frame.shape[:2]
        
        # Create alert banner
        banner_height = 60
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, h - banner_height),
            (w, h),
            self.colors['no_helmet'],
            -1
        )
        
        # Blend with original frame
        alpha = 0.8
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        # Draw alert text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 3
        
        (text_w, text_h), _ = cv2.getTextSize(
            message, font, font_scale, font_thickness
        )
        
        text_x = (w - text_w) // 2
        text_y = h - (banner_height - text_h) // 2
        
        cv2.putText(
            frame,
            message,
            (text_x, text_y),
            font,
            font_scale,
            self.colors['text'],
            font_thickness,
            cv2.LINE_AA
        )
        
        return frame
    
    def create_grid_view(
        self,
        frames: List[np.ndarray],
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Create grid view of multiple frames
        
        Args:
            frames: List of frames
            labels: Optional labels for each frame
            
        Returns:
            Combined grid image
        """
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        n_frames = len(frames)
        
        # Determine grid dimensions
        if n_frames == 1:
            return frames[0]
        elif n_frames <= 4:
            rows, cols = 2, 2
        elif n_frames <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 4
        
        # Resize frames to same size
        target_h = 480 // rows
        target_w = 640 // cols
        
        resized_frames = []
        for i, frame in enumerate(frames[:rows * cols]):
            resized = cv2.resize(frame, (target_w, target_h))
            
            # Add label if provided
            if labels and i < len(labels):
                cv2.putText(
                    resized,
                    labels[i],
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            resized_frames.append(resized)
        
        # Pad with black frames if needed
        while len(resized_frames) < rows * cols:
            resized_frames.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
        
        # Arrange in grid
        grid_rows = []
        for i in range(rows):
            row_frames = resized_frames[i * cols:(i + 1) * cols]
            grid_rows.append(np.hstack(row_frames))
        
        grid = np.vstack(grid_rows)
        
        return grid
    
    def add_timestamp(
        self,
        frame: np.ndarray,
        position: str = "top-right"
    ) -> np.ndarray:
        """
        Add timestamp to frame
        
        Args:
            frame: Input frame
            position: Position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            
        Returns:
            Frame with timestamp
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(
            timestamp, font, font_scale, font_thickness
        )
        
        h, w = frame.shape[:2]
        padding = 10
        
        # Calculate position
        if position == "top-left":
            x, y = padding, text_h + padding
        elif position == "top-right":
            x, y = w - text_w - padding, text_h + padding
        elif position == "bottom-left":
            x, y = padding, h - padding
        else:  # bottom-right
            x, y = w - text_w - padding, h - padding
        
        # Draw background
        cv2.rectangle(
            frame,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw timestamp
        cv2.putText(
            frame,
            timestamp,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )
        
        return frame