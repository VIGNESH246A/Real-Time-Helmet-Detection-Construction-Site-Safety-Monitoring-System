"""
Constants and enumerations for the Helmet Detection System
"""

from enum import Enum

# Color definitions (BGR format for OpenCV)
class Colors:
    """Color constants for visualization"""
    GREEN = (0, 255, 0)      # Helmet detected
    RED = (0, 0, 255)        # No helmet (violation)
    BLUE = (255, 0, 0)       # Person detected
    YELLOW = (0, 255, 255)   # Warning
    WHITE = (255, 255, 255)  # Text
    BLACK = (0, 0, 0)        # Background
    ORANGE = (0, 165, 255)   # Alert
    CYAN = (255, 255, 0)     # Info
    PURPLE = (255, 0, 255)   # Special


class DetectionClass(Enum):
    """Detection class enumeration"""
    HELMET = 0
    NO_HELMET = 1
    PERSON = 2


class ViolationType(Enum):
    """Violation type enumeration"""
    NO_HELMET = "no_helmet"
    RESTRICTED_AREA = "restricted_area"
    UNSAFE_BEHAVIOR = "unsafe_behavior"


class CameraStatus(Enum):
    """Camera status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
MIN_DETECTION_SIZE = 400  # Minimum bounding box area

# Video processing
DEFAULT_FPS = 30
DEFAULT_RESOLUTION = (1280, 720)
MAX_FRAME_BUFFER = 64

# Violation settings
VIOLATION_COOLDOWN_SECONDS = 5
MIN_VIOLATION_CONFIDENCE = 0.6

# File paths
DEFAULT_MODEL_PATH = "models/helmet_detector.pt"
DEFAULT_DB_PATH = "data/violations.db"
DEFAULT_SNAPSHOT_DIR = "outputs/snapshots"
DEFAULT_REPORTS_DIR = "outputs/reports"
DEFAULT_LOGS_DIR = "data/logs"

# Database
DB_SCHEMA_VERSION = "1.0"
MAX_RECORDS_PER_QUERY = 1000

# Report settings
REPORT_DATE_FORMAT = "%Y-%m-%d"
REPORT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
REPORT_FILENAME_FORMAT = "safety_report_{date}_{type}.{ext}"

# Dashboard
DASHBOARD_REFRESH_RATE = 1  # seconds
MAX_DISPLAY_VIOLATIONS = 100

# Performance
GPU_BATCH_SIZE = 4
CPU_BATCH_SIZE = 1
NUM_WORKERS = 2

# Alert messages
ALERT_MESSAGES = {
    "no_helmet": "⚠️ SAFETY VIOLATION: Worker without helmet detected!",
    "multiple_violations": "🚨 CRITICAL: Multiple safety violations detected!",
    "camera_error": "❌ ERROR: Camera connection lost!",
    "system_ready": "✅ System initialized and ready",
}

# Visualization settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
TEXT_PADDING = 5

# Image quality
SNAPSHOT_QUALITY = 95  # JPEG quality (0-100)
THUMBNAIL_SIZE = (320, 240)

# Tracking
MAX_TRACK_AGE = 30  # frames
MIN_TRACK_HITS = 3  # minimum detections to confirm track

# System
APP_NAME = "Helmet Detection System"
APP_VERSION = "1.0.0"
DEVELOPER = "Senior CV Engineer"

# Supported formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']

# Error messages
ERROR_MESSAGES = {
    "model_load_failed": "Failed to load YOLO model",
    "camera_init_failed": "Failed to initialize camera",
    "db_connection_failed": "Failed to connect to database",
    "invalid_config": "Invalid configuration file",
}