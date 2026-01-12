"""
SQLite database manager for violation logging
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ViolationDatabase:
    """
    Database manager for safety violations
    """
    
    def __init__(self, db_path: str = "data/violations.db"):
        """
        Initialize database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Database initialized: {db_path}")
    
    def _initialize_database(self) -> None:
        """Create database tables if they don't exist"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Violations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    camera_id TEXT,
                    violation_type TEXT,
                    confidence REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    snapshot_path TEXT,
                    metadata TEXT,
                    resolved BOOLEAN DEFAULT 0
                )
            ''')
            
            # Camera sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    frames_processed INTEGER DEFAULT 0,
                    violations_detected INTEGER DEFAULT 0
                )
            ''')
            
            # Daily statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    camera_id TEXT,
                    total_violations INTEGER DEFAULT 0,
                    helmet_compliant INTEGER DEFAULT 0,
                    no_helmet INTEGER DEFAULT 0,
                    compliance_rate REAL,
                    UNIQUE(date, camera_id)
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_timestamp 
                ON violations(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_camera 
                ON violations(camera_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_daily_stats_date 
                ON daily_stats(date)
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get database connection
        
        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def add_violation(
        self,
        camera_id: str,
        violation_type: str,
        confidence: float,
        bbox: List[int],
        snapshot_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a new violation record
        
        Args:
            camera_id: Camera identifier
            violation_type: Type of violation
            confidence: Detection confidence
            bbox: Bounding box [x1, y1, x2, y2]
            snapshot_path: Path to saved snapshot
            metadata: Additional metadata
            
        Returns:
            Violation ID
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO violations (
                    camera_id, violation_type, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    snapshot_path, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                camera_id,
                violation_type,
                confidence,
                bbox[0], bbox[1], bbox[2], bbox[3],
                snapshot_path,
                json.dumps(metadata) if metadata else None
            ))
            
            violation_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.debug(f"Violation added: ID {violation_id}")
            return violation_id
            
        except Exception as e:
            logger.error(f"Error adding violation: {e}")
            return -1
    
    def get_violations(
        self,
        camera_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get violations with optional filtering
        
        Args:
            camera_id: Filter by camera ID
            start_date: Start datetime
            end_date: End datetime
            limit: Maximum number of records
            
        Returns:
            List of violation dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM violations WHERE 1=1"
            params = []
            
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            violations = []
            for row in rows:
                violations.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'camera_id': row['camera_id'],
                    'violation_type': row['violation_type'],
                    'confidence': row['confidence'],
                    'bbox': [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']],
                    'snapshot_path': row['snapshot_path'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'resolved': bool(row['resolved'])
                })
            
            return violations
            
        except Exception as e:
            logger.error(f"Error getting violations: {e}")
            return []
    
    def get_violations_count(
        self,
        camera_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Get count of violations
        
        Args:
            camera_id: Filter by camera ID
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            Number of violations
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT COUNT(*) as count FROM violations WHERE 1=1"
            params = []
            
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            cursor.execute(query, params)
            count = cursor.fetchone()['count']
            conn.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting violations count: {e}")
            return 0
    
    def update_daily_stats(
        self,
        date: datetime,
        camera_id: str,
        violations: int,
        helmet_count: int,
        no_helmet_count: int
    ) -> None:
        """
        Update daily statistics
        
        Args:
            date: Date for stats
            camera_id: Camera identifier
            violations: Total violations
            helmet_count: Number with helmet
            no_helmet_count: Number without helmet
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            total = helmet_count + no_helmet_count
            compliance_rate = helmet_count / total if total > 0 else 1.0
            
            cursor.execute('''
                INSERT INTO daily_stats (
                    date, camera_id, total_violations,
                    helmet_compliant, no_helmet, compliance_rate
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, camera_id) DO UPDATE SET
                    total_violations = total_violations + excluded.total_violations,
                    helmet_compliant = helmet_compliant + excluded.helmet_compliant,
                    no_helmet = no_helmet + excluded.no_helmet,
                    compliance_rate = (helmet_compliant + excluded.helmet_compliant) * 1.0 / 
                                     (helmet_compliant + excluded.helmet_compliant + no_helmet + excluded.no_helmet)
            ''', (
                date.date().isoformat(),
                camera_id,
                violations,
                helmet_count,
                no_helmet_count,
                compliance_rate
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating daily stats: {e}")
    
    def get_daily_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        camera_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get daily statistics
        
        Args:
            start_date: Start date
            end_date: End date
            camera_id: Filter by camera ID
            
        Returns:
            List of daily statistics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM daily_stats WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.date().isoformat())
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.date().isoformat())
            
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            query += " ORDER BY date DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            stats = []
            for row in rows:
                stats.append(dict(row))
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting daily stats: {e}")
            return []
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """
        Delete records older than specified days
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted records
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                DELETE FROM violations 
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted {deleted_count} old violation records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0