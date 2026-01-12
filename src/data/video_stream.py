"""
Video stream manager for multi-source input handling
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Thread-safe video stream handler for webcam, IP camera, and video files
    """
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        resolution: Optional[Tuple[int, int]] = None,
        fps: int = 30,
        buffer_size: int = 64,
        name: str = "VideoStream"
    ):
        """
        Initialize video stream
        
        Args:
            source: Video source (0 for webcam, path for file, URL for IP cam)
            resolution: Desired resolution (width, height)
            fps: Target FPS
            buffer_size: Frame buffer size
            name: Stream name for logging
        """
        self.source = source
        self.resolution = resolution
        self.fps = fps
        self.buffer_size = buffer_size
        self.name = name
        
        # Video capture object
        self.stream = None
        
        # Threading
        self.stopped = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Stats
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = None
        
        # Initialize stream
        self._initialize_stream()
    
    def _initialize_stream(self) -> bool:
        """
        Initialize video capture
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.stream = cv2.VideoCapture(self.source)
            
            if not self.stream.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set properties if resolution specified
            if self.resolution:
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set FPS
            self.stream.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            actual_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.stream.get(cv2.CAP_PROP_FPS))
            
            logger.info(
                f"{self.name} initialized: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing stream: {e}")
            return False
    
    def start(self) -> 'VideoStream':
        """
        Start the threaded video stream
        
        Returns:
            Self for chaining
        """
        if self.thread is not None and self.thread.is_alive():
            logger.warning(f"{self.name} already started")
            return self
        
        self.stopped = False
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._update, name=self.name)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"{self.name} started")
        return self
    
    def _update(self) -> None:
        """
        Continuously read frames from stream (runs in separate thread)
        """
        while not self.stopped:
            if not self.stream.isOpened():
                logger.warning(f"{self.name} connection lost, attempting reconnect...")
                time.sleep(1)
                self._initialize_stream()
                continue
            
            ret, frame = self.stream.read()
            
            if not ret:
                logger.warning(f"{self.name} failed to read frame")
                time.sleep(0.01)
                continue
            
            # Try to add frame to queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                self.frame_count += 1
            else:
                # Drop frame if queue is full
                self.dropped_frames += 1
                try:
                    # Remove oldest frame
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass
        
        self.stream.release()
    
    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next available frame
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            (success, frame) tuple
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None
    
    def read_nowait(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame without waiting
        
        Returns:
            (success, frame) tuple
        """
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None
    
    def stop(self) -> None:
        """Stop the video stream"""
        self.stopped = True
        
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        if self.stream is not None:
            self.stream.release()
        
        logger.info(f"{self.name} stopped")
    
    def is_running(self) -> bool:
        """Check if stream is running"""
        return not self.stopped and self.thread is not None and self.thread.is_alive()
    
    def get_fps(self) -> float:
        """
        Calculate actual FPS
        
        Returns:
            Current FPS
        """
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def get_stats(self) -> dict:
        """
        Get stream statistics
        
        Returns:
            Dictionary with stats
        """
        return {
            'name': self.name,
            'source': self.source,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'fps': self.get_fps(),
            'queue_size': self.frame_queue.qsize(),
            'is_running': self.is_running(),
        }
    
    def restart(self) -> bool:
        """
        Restart the stream
        
        Returns:
            True if successful
        """
        logger.info(f"Restarting {self.name}...")
        self.stop()
        time.sleep(0.5)
        
        if self._initialize_stream():
            self.start()
            return True
        return False
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class MultiStreamManager:
    """Manage multiple video streams simultaneously"""
    
    def __init__(self):
        """Initialize multi-stream manager"""
        self.streams = {}
        logger.info("MultiStreamManager initialized")
    
    def add_stream(
        self,
        stream_id: str,
        source: Union[int, str],
        **kwargs
    ) -> bool:
        """
        Add a new video stream
        
        Args:
            stream_id: Unique stream identifier
            source: Video source
            **kwargs: Additional VideoStream arguments
            
        Returns:
            True if added successfully
        """
        if stream_id in self.streams:
            logger.warning(f"Stream {stream_id} already exists")
            return False
        
        try:
            stream = VideoStream(source, name=stream_id, **kwargs)
            stream.start()
            self.streams[stream_id] = stream
            logger.info(f"Added stream: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding stream {stream_id}: {e}")
            return False
    
    def remove_stream(self, stream_id: str) -> bool:
        """
        Remove a stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if removed successfully
        """
        if stream_id not in self.streams:
            logger.warning(f"Stream {stream_id} not found")
            return False
        
        self.streams[stream_id].stop()
        del self.streams[stream_id]
        logger.info(f"Removed stream: {stream_id}")
        return True
    
    def read_all(self) -> dict:
        """
        Read frames from all streams
        
        Returns:
            Dictionary {stream_id: frame}
        """
        frames = {}
        for stream_id, stream in self.streams.items():
            ret, frame = stream.read_nowait()
            if ret:
                frames[stream_id] = frame
        return frames
    
    def stop_all(self) -> None:
        """Stop all streams"""
        for stream in self.streams.values():
            stream.stop()
        self.streams.clear()
        logger.info("All streams stopped")