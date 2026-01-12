"""
Object tracking module using Simple Online and Realtime Tracking (SORT)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Simple Kalman Filter for tracking bounding boxes
    """
    
    def __init__(self):
        """Initialize Kalman filter"""
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.dt = 1.0
        self.state = np.zeros(8)
        self.covariance = np.eye(8) * 1000
        
        # Process noise
        self.process_noise = np.eye(8)
        self.process_noise[4:, 4:] *= 0.01
        
        # Measurement noise
        self.measurement_noise = np.eye(4) * 10
    
    def predict(self) -> np.ndarray:
        """
        Predict next state
        
        Returns:
            Predicted state
        """
        # State transition matrix
        F = np.eye(8)
        F[0, 4] = self.dt
        F[1, 5] = self.dt
        F[2, 6] = self.dt
        F[3, 7] = self.dt
        
        # Predict
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.process_noise
        
        return self.state[:4]
    
    def update(self, measurement: np.ndarray) -> None:
        """
        Update filter with measurement
        
        Args:
            measurement: Measured bounding box [x, y, w, h]
        """
        # Measurement matrix
        H = np.zeros((4, 8))
        H[:4, :4] = np.eye(4)
        
        # Innovation
        y = measurement - H @ self.state
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.measurement_noise
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        self.covariance = (np.eye(8) - K @ H) @ self.covariance


class Track:
    """
    Single object track
    """
    
    _id_counter = 0
    
    def __init__(self, bbox: np.ndarray, detection_data: Optional[Dict] = None):
        """
        Initialize track
        
        Args:
            bbox: Bounding box [x, y, w, h]
            detection_data: Additional detection information
        """
        self.id = Track._id_counter
        Track._id_counter += 1
        
        self.kf = KalmanFilter()
        self.kf.state[:4] = bbox
        
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        
        self.detection_data = detection_data or {}
        self.history = [bbox]
    
    def predict(self) -> np.ndarray:
        """
        Predict next position
        
        Returns:
            Predicted bounding box
        """
        self.age += 1
        self.time_since_update += 1
        
        predicted = self.kf.predict()
        return predicted
    
    def update(self, bbox: np.ndarray, detection_data: Optional[Dict] = None) -> None:
        """
        Update track with new detection
        
        Args:
            bbox: New bounding box
            detection_data: Detection information
        """
        self.kf.update(bbox)
        self.hits += 1
        self.time_since_update = 0
        
        if detection_data:
            self.detection_data = detection_data
        
        self.history.append(bbox)
        
        # Keep history limited
        if len(self.history) > 30:
            self.history.pop(0)
    
    def get_state(self) -> np.ndarray:
        """
        Get current state
        
        Returns:
            Current bounding box
        """
        return self.kf.state[:4]


class ObjectTracker:
    """
    Multi-object tracker using SORT-like algorithm
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track without update
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        
        logger.info("Object tracker initialized")
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries with 'bbox' key
            
        Returns:
            List of tracked detections with 'track_id'
        """
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        if detections:
            matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
            
            # Update matched tracks
            for track_idx, det_idx in matched:
                bbox = self._bbox_to_xywh(detections[det_idx]['bbox'])
                self.tracks[track_idx].update(bbox, detections[det_idx])
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                bbox = self._bbox_to_xywh(detections[det_idx]['bbox'])
                self.tracks.append(Track(bbox, detections[det_idx]))
        
        # Remove old tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_age
        ]
        
        # Return confirmed tracks
        tracked_detections = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                det = track.detection_data.copy()
                det['track_id'] = track.id
                det['bbox'] = self._xywh_to_bbox(track.get_state())
                tracked_detections.append(det)
        
        return tracked_detections
    
    def _match_detections(
        self,
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU
        
        Args:
            detections: List of detections
            
        Returns:
            (matched pairs, unmatched detections, unmatched tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            track_bbox = self._xywh_to_bbox(track.get_state())
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track_bbox, det['bbox'])
        
        # Hungarian algorithm (greedy matching for simplicity)
        matched = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))
        
        # Greedy matching
        while True:
            if iou_matrix.size == 0:
                break
            
            # Find best match
            max_iou_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            
            if max_iou < self.iou_threshold:
                break
            
            track_idx, det_idx = max_iou_idx
            
            # Add to matched
            matched.append((unmatched_tracks[track_idx], unmatched_dets[det_idx]))
            
            # Remove from unmatched
            iou_matrix = np.delete(iou_matrix, track_idx, axis=0)
            iou_matrix = np.delete(iou_matrix, det_idx, axis=1)
            
            del unmatched_tracks[track_idx]
            del unmatched_dets[det_idx]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _bbox_to_xywh(self, bbox: List[int]) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]"""
        return np.array([
            bbox[0],
            bbox[1],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1]
        ])
    
    def _xywh_to_bbox(self, xywh: np.ndarray) -> List[int]:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        return [
            int(xywh[0]),
            int(xywh[1]),
            int(xywh[0] + xywh[2]),
            int(xywh[1] + xywh[3])
        ]
    
    def reset(self) -> None:
        """Reset tracker"""
        self.tracks = []
        Track._id_counter = 0
        logger.info("Tracker reset")
    
    def get_statistics(self) -> Dict:
        """
        Get tracker statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'active_tracks': len(self.tracks),
            'confirmed_tracks': len([t for t in self.tracks if t.hits >= self.min_hits]),
            'total_tracks_created': Track._id_counter
        }