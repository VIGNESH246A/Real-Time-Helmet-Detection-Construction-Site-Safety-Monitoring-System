"""
Unit tests for database operations
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import ViolationDatabase


class TestViolationDatabase:
    """Test cases for ViolationDatabase class"""
    
    @pytest.fixture
    def db(self, tmp_path):
        """Create temporary database"""
        db_path = tmp_path / "test_violations.db"
        return ViolationDatabase(str(db_path))
    
    def test_database_initialization(self, db):
        """Test database initialization"""
        assert db is not None
        assert Path(db.db_path).exists()
    
    def test_add_violation(self, db):
        """Test adding violation"""
        violation_id = db.add_violation(
            camera_id="CAM_001",
            violation_type="no_helmet",
            confidence=0.85,
            bbox=[10, 20, 100, 200],
            snapshot_path="/path/to/snapshot.jpg",
            metadata={'severity': 'high'}
        )
        
        assert violation_id > 0
    
    def test_get_violations(self, db):
        """Test retrieving violations"""
        # Add some violations
        for i in range(5):
            db.add_violation(
                camera_id=f"CAM_{i:03d}",
                violation_type="no_helmet",
                confidence=0.8 + i * 0.02,
                bbox=[10*i, 20*i, 100*i, 200*i]
            )
        
        violations = db.get_violations(limit=10)
        
        assert len(violations) == 5
        assert all('id' in v for v in violations)
        assert all('bbox' in v for v in violations)
    
    def test_get_violations_by_camera(self, db):
        """Test filtering violations by camera"""
        # Add violations for different cameras
        db.add_violation("CAM_001", "no_helmet", 0.8, [10, 20, 100, 200])
        db.add_violation("CAM_002", "no_helmet", 0.7, [10, 20, 100, 200])
        db.add_violation("CAM_001", "no_helmet", 0.9, [10, 20, 100, 200])
        
        violations = db.get_violations(camera_id="CAM_001")
        
        assert len(violations) == 2
        assert all(v['camera_id'] == "CAM_001" for v in violations)
    
    def test_get_violations_count(self, db):
        """Test counting violations"""
        # Add violations
        for i in range(10):
            db.add_violation("CAM_001", "no_helmet", 0.8, [10, 20, 100, 200])
        
        count = db.get_violations_count(camera_id="CAM_001")
        
        assert count == 10
    
    def test_update_daily_stats(self, db):
        """Test updating daily statistics"""
        db.update_daily_stats(
            date=datetime.now(),
            camera_id="CAM_001",
            violations=5,
            helmet_count=10,
            no_helmet_count=5
        )
        
        stats = db.get_daily_stats()
        
        assert len(stats) > 0
        assert stats[0]['total_violations'] == 5
    
    def test_cleanup_old_records(self, db):
        """Test cleaning up old records"""
        # Add some violations
        for i in range(5):
            db.add_violation("CAM_001", "no_helmet", 0.8, [10, 20, 100, 200])
        
        # Cleanup (should delete nothing since records are new)
        deleted = db.cleanup_old_records(days=30)
        
        assert deleted == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])