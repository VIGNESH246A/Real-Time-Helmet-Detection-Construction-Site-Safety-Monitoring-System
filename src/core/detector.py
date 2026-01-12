"""
YOLO-based helmet detection engine
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HelmetDetector:
    """
    YOLOv8-based helmet detection system
    """
    
    def __init__(
        self,
        model_path: str = "models/helmet_detector.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        input_size: int = 640
    ):
        """
        Initialize helmet detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda', 'cpu', 'mps')
            input_size: Model input size
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.input_size = input_size
        
        # Class names
        self.class_names = {
            0: "helmet",
            1: "no-helmet",
            2: "person"
        }
        
        # Load model
        self.model = None
        self._load_model()
        
        # Statistics
        self.total_inferences = 0
        self.total_detections = 0
    
    def _load_model(self) -> bool:
        """
        Load YOLO model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if custom model exists
            if not Path(self.model_path).exists():
                logger.warning(f"Custom model not found at {self.model_path}")
                logger.info("Loading pre-trained YOLOv8n model...")
                self.model = YOLO("yolov8n.pt")
            else:
                self.model = YOLO(self.model_path)
            
            # Set device
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU")
                self.device = "cpu"
            
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model classes: {self.model.names if hasattr(self.model, 'names') else self.class_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = False
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Perform detection on image
        
        Args:
            image: Input BGR image
            return_crops: Whether to return cropped detections
            
        Returns:
            (detections, annotated_image) tuple
            detections: List of dicts with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
                - crop: np.ndarray (if return_crops=True)
        """
        if image is None or image.size == 0:
            logger.warning("Empty image received")
            return [], image
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False,
                device=self.device
            )
            
            self.total_inferences += 1
            
            # Parse results
            detections = []
            annotated_image = image.copy()
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Extract box data
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Get class name
                        if hasattr(self.model, 'names'):
                            cls_name = self.model.names[cls_id]
                        else:
                            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        
                        # Create detection dict
                        detection = {
                            'bbox': xyxy.astype(int).tolist(),
                            'confidence': conf,
                            'class_id': cls_id,
                            'class_name': cls_name
                        }
                        
                        # Add crop if requested
                        if return_crops:
                            x1, y1, x2, y2 = detection['bbox']
                            crop = image[y1:y2, x1:x2]
                            detection['crop'] = crop
                        
                        detections.append(detection)
                        self.total_detections += 1
                
                # Get annotated image
                if hasattr(result, 'plot'):
                    annotated_image = result.plot()
            
            return detections, annotated_image
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return [], image
    
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[List[Dict]]:
        """
        Batch detection for multiple images
        
        Args:
            images: List of BGR images
            
        Returns:
            List of detection lists
        """
        try:
            results = self.model.predict(
                images,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False,
                device=self.device
            )
            
            all_detections = []
            
            for result in results:
                detections = []
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        
                        detections.append({
                            'bbox': xyxy.astype(int).tolist(),
                            'confidence': conf,
                            'class_id': cls_id,
                            'class_name': cls_name
                        })
                
                all_detections.append(detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Error in batch detection: {e}")
            return [[] for _ in images]
    
    def classify_helmet_status(self, detections: List[Dict]) -> Dict:
        """
        Classify overall helmet compliance status
        
        Args:
            detections: List of detections
            
        Returns:
            Status dictionary with counts and violations
        """
        helmet_count = 0
        no_helmet_count = 0
        violations = []
        
        for det in detections:
            class_name = det['class_name'].lower()
            
            if 'helmet' in class_name and 'no' not in class_name:
                helmet_count += 1
            elif 'no-helmet' in class_name or 'no_helmet' in class_name:
                no_helmet_count += 1
                violations.append(det)
        
        status = {
            'helmet_count': helmet_count,
            'no_helmet_count': no_helmet_count,
            'total_people': helmet_count + no_helmet_count,
            'compliance_rate': helmet_count / (helmet_count + no_helmet_count) if (helmet_count + no_helmet_count) > 0 else 1.0,
            'has_violations': no_helmet_count > 0,
            'violations': violations
        }
        
        return status
    
    def annotate_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Annotate image with detection boxes
        
        Args:
            image: Input image
            detections: List of detections
            show_confidence: Whether to show confidence scores
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Color based on class
            if 'helmet' in class_name.lower() and 'no' not in class_name.lower():
                color = (0, 255, 0)  # Green for helmet
            elif 'no' in class_name.lower():
                color = (0, 0, 255)  # Red for no helmet
            else:
                color = (255, 0, 0)  # Blue for person
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            if show_confidence:
                label = f"{class_name}: {conf:.2f}"
            else:
                label = class_name
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def get_stats(self) -> Dict:
        """
        Get detector statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_inferences': self.total_inferences,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / self.total_inferences if self.total_inferences > 0 else 0,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
        }