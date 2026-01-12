"""
Advanced logging utilities for the system
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors"""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class SystemLogger:
    """
    Centralized logging system
    """
    
    def __init__(
        self,
        name: str = "HelmetDetection",
        log_dir: str = "data/logs",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize system logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (rotating)
        log_file = self.log_dir / f"{name.lower()}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_file = self.log_dir / f"{name.lower()}_errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        self.logger.info(f"Logger initialized: {name}")
    
    def get_logger(self) -> logging.Logger:
        """
        Get logger instance
        
        Returns:
            Logger instance
        """
        return self.logger
    
    def log_system_info(self) -> None:
        """Log system information"""
        import platform
        import torch
        
        self.logger.info("=" * 70)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 70)
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Python: {platform.python_version()}")
        self.logger.info(f"PyTorch: {torch.__version__}")
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.logger.info("=" * 70)
    
    def create_session_log(self, session_name: Optional[str] = None) -> str:
        """
        Create a new session-specific log file
        
        Args:
            session_name: Name for the session
            
        Returns:
            Path to session log file
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session_file = self.log_dir / f"session_{session_name}.log"
        
        session_handler = logging.FileHandler(session_file)
        session_handler.setLevel(logging.DEBUG)
        session_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        session_handler.setFormatter(session_formatter)
        self.logger.addHandler(session_handler)
        
        self.logger.info(f"Session log created: {session_file}")
        return str(session_file)


def setup_logging(
    log_dir: str = "data/logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG"
) -> logging.Logger:
    """
    Quick setup for logging
    
    Args:
        log_dir: Log directory
        console_level: Console log level
        file_level: File log level
        
    Returns:
        Configured logger
    """
    system_logger = SystemLogger(
        name="HelmetDetection",
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level
    )
    
    return system_logger.get_logger()


# Performance logging decorator
def log_performance(func):
    """Decorator to log function execution time"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("HelmetDetection")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {elapsed:.4f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {e}")
            raise
    
    return wrapper