"""
Violation detection and management engine
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ViolationEngine:
    """
    Safety violation detection and tracking engine
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        cooldown_seconds: int = 5,
        snapshot_dir: str = "outputs/snapshots",
        snapshot_quality: int = 95
    ):
        """
        Initialize violation engine
        
        Args:
            min_confidence: Minimum confidence for violation
            cooldown_seconds: Cooldown period between same violations
            snapshot_dir: Directory to save violation snapshots
            snapshot_quality: JPEG quality for snapshots (0-100)
        """
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self.snapshot_dir = snapshot_dir
        self.snapshot_quality = snapshot_quality
        
        # Track recent violations to avoid duplicates
        self.recent_violations = {}
        
        # Statistics
        self.total_violations = 0
        self.violations_by_type = {}
        
        # Ensure snapshot directory exists
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Violation engine initialized")
    
    def check_violations(
        self,
        detections: List[Dict],
        camera_id: str = "default"
    ) -> List[Dict]:
        """
        Check detections for safety violations
        
        Args:
            detections: List of detection dictionaries
            camera_id: Camera identifier
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        current_time = datetime.now()
        
        for det in detections:
            class_name = det['class_name'].lower()
            confidence = det['confidence']
            
            # Check if this is a no-helmet violation
            if 'no-helmet' in class_name or 'no_helmet' in class_name:
                if confidence >= self.min_confidence:
                    # Check cooldown
                    if self._is_in_cooldown(det, camera_id, current_time):
                        continue
                    
                    # Create violation record
                    violation = {
                        'type': 'no_helmet',
                        'timestamp': current_time,
                        'camera_id': camera_id,
                        'detection': det,
                        'severity': self._calculate_severity(confidence)
                    }
                    
                    violations.append(violation)
                    
                    # Update tracking
                    self._add_to_recent_violations(det, camera_id, current_time)
                    
                    # Update statistics
                    self.total_violations += 1
                    self.violations_by_type['no_helmet'] = \
                        self.violations_by_type.get('no_helmet', 0) + 1
        
        return violations
    
    def _is_in_cooldown(
        self,
        detection: Dict,
        camera_id: str,
        current_time: datetime
    ) -> bool:
        """
        Check if detection is in cooldown period
        
        Args:
            detection: Detection dictionary
            camera_id: Camera identifier
            current_time: Current timestamp
            
        Returns:
            True if in cooldown, False otherwise
        """
        key = f"{camera_id}_{self._get_detection_key(detection)}"
        
        if key in self.recent_violations:
            last_time = self.recent_violations[key]
            time_diff = (current_time - last_time).total_seconds()
            
            if time_diff < self.cooldown_seconds:
                return True
        
        return False
    
    def _add_to_recent_violations(
        self,
        detection: Dict,
        camera_id: str,
        timestamp: datetime
    ) -> None:
        """
        Add detection to recent violations tracking
        
        Args:
            detection: Detection dictionary
            camera_id: Camera identifier
            timestamp: Timestamp
        """
        key = f"{camera_id}_{self._get_detection_key(detection)}"
        self.recent_violations[key] = timestamp
        
        # Clean up old entries
        self._cleanup_old_violations(timestamp)
    
    def _get_detection_key(self, detection: Dict) -> str:
        """
        Generate unique key for detection based on location
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Unique key string
        """
        bbox = detection['bbox']
        # Use center point and approximate size
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        
        # Quantize to reduce sensitivity
        cx_q = (cx // 50) * 50
        cy_q = (cy // 50) * 50
        size_q = (size // 50) * 50
        
        return f"{cx_q}_{cy_q}_{size_q}"
    
    def _cleanup_old_violations(self, current_time: datetime) -> None:
        """
        Remove old violations from tracking
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - timedelta(seconds=self.cooldown_seconds * 2)
        
        keys_to_remove = [
            key for key, timestamp in self.recent_violations.items()
            if timestamp < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self.recent_violations[key]
    
    def _calculate_severity(self, confidence: float) -> str:
        """
        Calculate violation severity based on confidence
        
        Args:
            confidence: Detection confidence
            
        Returns:
            Severity level: 'low', 'medium', 'high', 'critical'
        """
        if confidence >= 0.9:
            return 'critical'
        elif confidence >= 0.75:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def save_violation_snapshot(
        self,
        image: np.ndarray,
        violation: Dict,
        annotate: bool = True
    ) -> Optional[str]:
        """
        Save snapshot image for violation
        
        Args:
            image: Original frame
            violation: Violation dictionary
            annotate: Whether to draw bounding box
            
        Returns:
            Path to saved snapshot, or None if failed
        """
        try:
            snapshot = image.copy()
            
            # Annotate if requested
            if annotate:
                det = violation['detection']
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw red box
                cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Add label
                label = f"VIOLATION: {det['class_name']} ({det['confidence']:.2f})"
                cv2.putText(
                    snapshot,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                # Add timestamp
                timestamp_str = violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(
                    snapshot,
                    timestamp_str,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Generate filename
            timestamp_str = violation['timestamp'].strftime('%Y%m%d_%H%M%S')
            camera_id = violation['camera_id']
            filename = f"violation_{camera_id}_{timestamp_str}.jpg"
            filepath = Path(self.snapshot_dir) / filename
            
            # Save image
            cv2.imwrite(
                str(filepath),
                snapshot,
                [cv2.IMWRITE_JPEG_QUALITY, self.snapshot_quality]
            )
            
            logger.info(f"Violation snapshot saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving violation snapshot: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """
        Get violation statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_violations': self.total_violations,
            'violations_by_type': self.violations_by_type.copy(),
            'active_cooldowns': len(self.recent_violations)
        }
    
    def reset_statistics(self) -> None:
        """Reset violation statistics"""
        self.total_violations = 0
        self.violations_by_type = {}
        logger.info("Violation statistics reset")
    
    def calculate_compliance_rate(
        self,
        helmet_count: int,
        no_helmet_count: int
    ) -> float:
        """
        Calculate helmet compliance rate
        
        Args:
            helmet_count: Number of people with helmets
            no_helmet_count: Number of people without helmets
            
        Returns:
            Compliance rate (0.0 to 1.0)
        """
        total = helmet_count + no_helmet_count
        return helmet_count / total if total > 0 else 1.0
    
    def get_compliance_status(self, compliance_rate: float) -> Dict:
        """
        Get compliance status based on rate
        
        Args:
            compliance_rate: Compliance rate (0.0 to 1.0)
            
        Returns:
            Status dictionary with level and message
        """
        if compliance_rate >= 0.95:
            return {
                'level': 'excellent',
                'message': 'Excellent compliance!',
                'color': 'green'
            }
        elif compliance_rate >= 0.85:
            return {
                'level': 'good',
                'message': 'Good compliance',
                'color': 'lightgreen'
            }
        elif compliance_rate >= 0.70:
            return {
                'level': 'fair',
                'message': 'Fair compliance - improvement needed',
                'color': 'yellow'
            }
        elif compliance_rate >= 0.50:
            return {
                'level': 'poor',
                'message': 'Poor compliance - immediate action required',
                'color': 'orange'
            }
        else:
            return {
                'level': 'critical',
                'message': 'CRITICAL: Severe non-compliance!',
                'color': 'red'
            }