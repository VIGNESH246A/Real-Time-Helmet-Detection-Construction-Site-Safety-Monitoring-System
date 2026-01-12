"""
Main application entry point for Helmet Detection System
"""

import cv2
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Import custom modules
from src.utils.config_loader import get_config
from src.utils.helpers import create_timestamp
from src.core.detector import HelmetDetector
from src.core.preprocessor import ImagePreprocessor
from src.data.video_stream import VideoStream
from src.data.database import ViolationDatabase
from src.analysis.violation_engine import ViolationEngine
from src.analysis.alert_manager import AlertManager
from src.analysis.report_generator import ReportGenerator
from src.ui.visualizer import FrameVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class HelmetDetectionSystem:
    """
    Main helmet detection and safety monitoring system
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the system
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("=" * 70)
        logger.info("Initializing Helmet Detection System")
        logger.info("=" * 70)
        
        # Load configuration
        self.config = get_config(config_path)
        self.config.ensure_directories()
        
        # Initialize components
        self._initialize_components()
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        self.running = False
        
        logger.info("System initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all system components"""
        
        # Image preprocessor
        self.preprocessor = ImagePreprocessor(
            enable_low_light=self.config.get('preprocessing.enable_low_light_enhancement', True),
            enable_noise_reduction=self.config.get('preprocessing.enable_noise_reduction', True)
        )
        
        # Helmet detector
        self.detector = HelmetDetector(
            model_path=self.config.get('model.weights_path', 'models/helmet_detector.pt'),
            confidence_threshold=self.config.get('model.confidence_threshold', 0.5),
            iou_threshold=self.config.get('model.iou_threshold', 0.45),
            device=self.config.get('model.device', 'cuda'),
            input_size=self.config.get('model.input_size', 640)
        )
        
        # Video stream
        video_source = self.config.get('video.default_source', 0)
        self.stream = VideoStream(
            source=video_source,
            resolution=(
                self.config.get('video.resolution.width', 1280),
                self.config.get('video.resolution.height', 720)
            ),
            fps=self.config.get('video.fps', 30),
            buffer_size=self.config.get('video.buffer_size', 64)
        )
        
        # Database
        self.database = ViolationDatabase(
            db_path=self.config.get('storage.database_path', 'data/violations.db')
        )
        
        # Violation engine
        self.violation_engine = ViolationEngine(
            min_confidence=self.config.get('violation.min_confidence', 0.6),
            cooldown_seconds=self.config.get('violation.cooldown_seconds', 5),
            snapshot_dir=self.config.get('storage.snapshots_dir', 'outputs/snapshots')
        )
        
        # Alert manager
        self.alert_manager = AlertManager(
            enable_visual=self.config.get('alerts.visual_enabled', True),
            enable_audio=self.config.get('alerts.audio_enabled', False),
            alert_sound_path=self.config.get('violation.alert_sound_path', None)
        )
        
        # Frame visualizer
        self.visualizer = FrameVisualizer(
            show_confidence=True,
            show_fps=self.config.get('system.display_fps', True),
            show_stats=True
        )
        
        # Report generator
        self.report_generator = ReportGenerator(
            output_dir=self.config.get('storage.reports_dir', 'outputs/reports')
        )
    
    def run(self, display: bool = True, record: bool = False) -> None:
        """
        Run the detection system
        
        Args:
            display: Whether to display video feed
            record: Whether to record output video
        """
        logger.info("Starting detection system...")
        
        self.running = True
        self.start_time = time.time()
        
        # Start video stream
        self.stream.start()
        time.sleep(1.0)  # Allow camera to warm up
        
        # Video writer setup
        video_writer = None
        if record:
            video_writer = self._setup_video_writer()
        
        # FPS calculation
        fps = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.stream.read(timeout=2.0)
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    continue
                
                # Preprocess frame
                processed_frame = self.preprocessor.preprocess(frame)
                
                # Detect helmets
                detections, annotated_frame = self.detector.detect(processed_frame)
                
                # Classify helmet status
                status = self.detector.classify_helmet_status(detections)
                
                # Check for violations
                violations = self.violation_engine.check_violations(
                    detections,
                    camera_id="CAM_001"
                )
                
                # Handle violations
                if violations:
                    self._handle_violations(frame, violations, status)
                
                # Annotate frame
                display_frame = self.visualizer.annotate_frame(
                    frame,
                    detections,
                    fps=fps,
                    stats=status
                )
                
                # Add timestamp
                display_frame = self.visualizer.add_timestamp(display_frame)
                
                # Show violation alert if needed
                if status['has_violations']:
                    display_frame = self.visualizer.draw_violation_alert(display_frame)
                
                # Update statistics
                self.frame_count += 1
                fps_frame_count += 1
                
                # Calculate FPS every second
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Display frame
                if display:
                    cv2.imshow('Helmet Detection System', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit signal received")
                        break
                    elif key == ord('s'):
                        self._save_screenshot(display_frame)
                    elif key == ord('r'):
                        self._generate_report()
                
                # Write to video file
                if video_writer:
                    video_writer.write(display_frame)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        finally:
            self._cleanup(video_writer)
    
    def _handle_violations(
        self,
        frame,
        violations: list,
        status: dict
    ) -> None:
        """
        Handle detected violations
        
        Args:
            frame: Current frame
            violations: List of violations
            status: Detection status
        """
        for violation in violations:
            # Save snapshot
            if self.config.get('violation.snapshot_enabled', True):
                snapshot_path = self.violation_engine.save_violation_snapshot(
                    frame, violation, annotate=True
                )
                violation['snapshot_path'] = snapshot_path
            
            # Trigger alert
            severity = violation.get('severity', 'medium')
            self.alert_manager.trigger_alert(violation, severity)
            
            # Log to database
            det = violation['detection']
            self.database.add_violation(
                camera_id=violation['camera_id'],
                violation_type=violation['type'],
                confidence=det['confidence'],
                bbox=det['bbox'],
                snapshot_path=violation.get('snapshot_path'),
                metadata={
                    'severity': severity,
                    'class_name': det['class_name']
                }
            )
            
            logger.warning(f"Violation detected: {violation['type']} at {violation['timestamp']}")
        
        # Update daily statistics
        self.database.update_daily_stats(
            date=datetime.now(),
            camera_id="CAM_001",
            violations=len(violations),
            helmet_count=status['helmet_count'],
            no_helmet_count=status['no_helmet_count']
        )
    
    def _setup_video_writer(self):
        """Setup video writer for recording"""
        try:
            timestamp = create_timestamp()
            filename = f"recording_{timestamp}.mp4"
            filepath = Path(self.config.get('storage.videos_dir', 'outputs/videos')) / filename
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.config.get('video.fps', 30)
            width = self.config.get('video.resolution.width', 1280)
            height = self.config.get('video.resolution.height', 720)
            
            writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
            logger.info(f"Recording to: {filepath}")
            return writer
            
        except Exception as e:
            logger.error(f"Failed to setup video writer: {e}")
            return None
    
    def _save_screenshot(self, frame):
        """Save screenshot of current frame"""
        try:
            timestamp = create_timestamp()
            filename = f"screenshot_{timestamp}.jpg"
            filepath = Path(self.config.get('storage.snapshots_dir', 'outputs/snapshots')) / filename
            cv2.imwrite(str(filepath), frame)
            logger.info(f"Screenshot saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
    
    def _generate_report(self):
        """Generate safety report on demand"""
        try:
            logger.info("Generating report...")
            violations = self.database.get_violations(limit=1000)
            
            stats = {
                'total_violations': len(violations),
                'helmet_count': 0,
                'no_helmet_count': len(violations),
                'compliance_rate': 0.0
            }
            
            report_path = self.report_generator.generate_daily_report(
                date=datetime.now(),
                violations=violations,
                stats=stats,
                format='pdf'
            )
            
            logger.info(f"Report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    def _cleanup(self, video_writer):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        self.running = False
        
        if video_writer:
            video_writer.release()
        
        self.stream.stop()
        cv2.destroyAllWindows()
        
        # Print statistics
        runtime = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Runtime: {runtime:.2f} seconds")
        logger.info(f"Average FPS: {self.frame_count / runtime if runtime > 0 else 0:.2f}")
        
        logger.info("System shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Helmet Detection System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without video display')
    parser.add_argument('--record', action='store_true',
                        help='Record output video')
    parser.add_argument('--source', type=str, default=None,
                        help='Video source (camera index or file path)')
    
    args = parser.parse_args()
    
    # Create and run system
    system = HelmetDetectionSystem(config_path=args.config)
    
    # Override source if specified
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        system.stream.source = source
        system.stream.restart()
    
    system.run(display=not args.no_display, record=args.record)


if __name__ == "__main__":
    main()