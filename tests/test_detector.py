"""
Unit tests for helmet detector
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.detector import HelmetDetector


class TestHelmetDetector:
    """Test cases for HelmetDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return HelmetDetector(
            model_path="yolov8n.pt",  # Use base model for testing
            confidence_threshold=0.5,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image"""
        # Create dummy image (640x640x3)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        return image
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.model is not None
        assert detector.confidence_threshold == 0.5
    
    def test_detect_on_image(self, detector, sample_image):
        """Test detection on sample image"""
        detections, annotated = detector.detect(sample_image)
        
        assert isinstance(detections, list)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == sample_image.shape
    
    def test_detect_empty_image(self, detector):
        """Test detection on empty image"""
        empty_image = None
        detections, annotated = detector.detect(empty_image)
        
        assert detections == []
        assert annotated is None
    
    def test_classify_helmet_status(self, detector):
        """Test helmet status classification"""
        # Mock detections
        detections = [
            {'class_name': 'helmet', 'confidence': 0.8},
            {'class_name': 'helmet', 'confidence': 0.9},
            {'class_name': 'no-helmet', 'confidence': 0.7}
        ]
        
        status = detector.classify_helmet_status(detections)
        
        assert status['helmet_count'] == 2
        assert status['no_helmet_count'] == 1
        assert status['total_people'] == 3
        assert status['has_violations'] == True
    
    def test_annotate_image(self, detector, sample_image):
        """Test image annotation"""
        detections = [
            {
                'bbox': [10, 10, 100, 100],
                'confidence': 0.8,
                'class_name': 'helmet'
            }
        ]
        
        annotated = detector.annotate_image(sample_image, detections)
        
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == sample_image.shape
    
    def test_get_stats(self, detector):
        """Test statistics retrieval"""
        stats = detector.get_stats()
        
        assert 'total_inferences' in stats
        assert 'total_detections' in stats
        assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])