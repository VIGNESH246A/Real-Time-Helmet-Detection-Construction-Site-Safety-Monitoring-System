"""
Configuration loader and manager
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage system configuration"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}")
                self._create_default_config()
                return
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default configuration"""
        self.config = {
            'model': {
                'name': 'yolov8n',
                'weights_path': 'models/helmet_detector.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'device': 'cuda',
            },
            'video': {
                'default_source': 0,
                'fps': 30,
                'frame_skip': 2,
            },
            'violation': {
                'min_confidence': 0.6,
                'cooldown_seconds': 5,
                'snapshot_enabled': True,
            },
            'storage': {
                'database_path': 'data/violations.db',
                'snapshots_dir': 'outputs/snapshots',
                'reports_dir': 'outputs/reports',
            }
        }
        logger.info("Using default configuration")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots)
        
        Args:
            key: Configuration key (e.g., 'model.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports nested keys with dots)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file
        
        Args:
            path: Path to save config (uses original path if None)
        """
        save_path = path or self.config_path
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if valid, False otherwise
        """
        required_keys = [
            'model.weights_path',
            'video.default_source',
            'storage.database_path',
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Missing required configuration: {key}")
                return False
        
        return True
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config.copy()
    
    def load_camera_config(self, path: str = "config/camera_config.json") -> Dict:
        """
        Load camera-specific configuration
        
        Args:
            path: Path to camera config JSON
            
        Returns:
            Camera configuration dictionary
        """
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading camera config: {e}")
        
        return {}
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist"""
        dirs = [
            self.get('storage.snapshots_dir', 'outputs/snapshots'),
            self.get('storage.reports_dir', 'outputs/reports'),
            self.get('storage.logs_dir', 'data/logs'),
            'models',
            'data',
        ]
        
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


# Singleton instance
_config_instance = None

def get_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Get singleton configuration instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    
    return _config_instance