"""
Alert and notification management system
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manage alerts and notifications for safety violations
    """
    
    def __init__(
        self,
        enable_visual: bool = True,
        enable_audio: bool = False,
        enable_logging: bool = True,
        alert_sound_path: Optional[str] = None
    ):
        """
        Initialize alert manager
        
        Args:
            enable_visual: Enable visual alerts
            enable_audio: Enable audio alerts
            enable_logging: Enable alert logging
            alert_sound_path: Path to alert sound file
        """
        self.enable_visual = enable_visual
        self.enable_audio = enable_audio
        self.enable_logging = enable_logging
        self.alert_sound_path = alert_sound_path
        
        # Alert history
        self.alert_history = []
        self.max_history_size = 1000
        
        # Statistics
        self.total_alerts = 0
        self.alerts_by_type = {}
        
        # Callbacks
        self.alert_callbacks = []
        
        # Audio support
        self.audio_player = None
        if enable_audio:
            self._initialize_audio()
        
        logger.info("Alert manager initialized")
    
    def _initialize_audio(self) -> None:
        """Initialize audio playback for alerts"""
        try:
            import pygame
            pygame.mixer.init()
            self.audio_player = pygame.mixer
            logger.info("Audio alerts enabled")
        except Exception as e:
            logger.warning(f"Could not initialize audio: {e}")
            self.enable_audio = False
    
    def trigger_alert(
        self,
        violation: Dict,
        severity: str = "medium"
    ) -> None:
        """
        Trigger an alert for a violation
        
        Args:
            violation: Violation dictionary
            severity: Alert severity ('low', 'medium', 'high', 'critical')
        """
        alert = {
            'timestamp': datetime.now(),
            'violation': violation,
            'severity': severity,
            'message': self._generate_alert_message(violation)
        }
        
        # Visual alert
        if self.enable_visual:
            self._show_visual_alert(alert)
        
        # Audio alert
        if self.enable_audio:
            self._play_audio_alert(severity)
        
        # Log alert
        if self.enable_logging:
            self._log_alert(alert)
        
        # Store in history
        self._add_to_history(alert)
        
        # Execute callbacks
        self._execute_callbacks(alert)
        
        # Update statistics
        self.total_alerts += 1
        violation_type = violation.get('type', 'unknown')
        self.alerts_by_type[violation_type] = \
            self.alerts_by_type.get(violation_type, 0) + 1
        
        logger.info(f"Alert triggered: {alert['message']}")
    
    def _generate_alert_message(self, violation: Dict) -> str:
        """
        Generate human-readable alert message
        
        Args:
            violation: Violation dictionary
            
        Returns:
            Alert message string
        """
        violation_type = violation.get('type', 'unknown')
        camera_id = violation.get('camera_id', 'unknown')
        timestamp = violation.get('timestamp', datetime.now())
        
        if violation_type == 'no_helmet':
            return f"⚠️ SAFETY VIOLATION: Worker without helmet detected on {camera_id} at {timestamp.strftime('%H:%M:%S')}"
        else:
            return f"⚠️ VIOLATION: {violation_type} detected on {camera_id}"
    
    def _show_visual_alert(self, alert: Dict) -> None:
        """
        Display visual alert (console output)
        
        Args:
            alert: Alert dictionary
        """
        severity = alert['severity']
        message = alert['message']
        
        # Color codes for terminal
        colors = {
            'low': '\033[93m',      # Yellow
            'medium': '\033[93m',   # Yellow
            'high': '\033[91m',     # Red
            'critical': '\033[91m'  # Red
        }
        reset = '\033[0m'
        
        color = colors.get(severity, '')
        print(f"\n{color}{'=' * 70}")
        print(f"{message}")
        print(f"{'=' * 70}{reset}\n")
    
    def _play_audio_alert(self, severity: str) -> None:
        """
        Play audio alert sound
        
        Args:
            severity: Alert severity
        """
        if not self.audio_player or not self.alert_sound_path:
            return
        
        try:
            if Path(self.alert_sound_path).exists():
                # Play in separate thread to avoid blocking
                def play_sound():
                    try:
                        self.audio_player.music.load(self.alert_sound_path)
                        self.audio_player.music.play()
                    except Exception as e:
                        logger.error(f"Error playing alert sound: {e}")
                
                thread = threading.Thread(target=play_sound, daemon=True)
                thread.start()
            else:
                # Generate simple beep as fallback
                print('\a')  # System beep
                
        except Exception as e:
            logger.error(f"Error in audio alert: {e}")
    
    def _log_alert(self, alert: Dict) -> None:
        """
        Log alert to file
        
        Args:
            alert: Alert dictionary
        """
        log_message = f"[{alert['timestamp']}] [{alert['severity'].upper()}] {alert['message']}"
        logger.warning(log_message)
    
    def _add_to_history(self, alert: Dict) -> None:
        """
        Add alert to history with size limit
        
        Args:
            alert: Alert dictionary
        """
        self.alert_history.append(alert)
        
        # Trim history if too large
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
    
    def _execute_callbacks(self, alert: Dict) -> None:
        """
        Execute registered callback functions
        
        Args:
            alert: Alert dictionary
        """
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error executing alert callback: {e}")
    
    def register_callback(self, callback: Callable) -> None:
        """
        Register a callback function to be called on alerts
        
        Args:
            callback: Function that accepts alert dictionary
        """
        self.alert_callbacks.append(callback)
        logger.info(f"Registered alert callback: {callback.__name__}")
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """
        Get most recent alerts
        
        Args:
            count: Number of alerts to retrieve
            
        Returns:
            List of alert dictionaries
        """
        return self.alert_history[-count:]
    
    def get_alerts_by_severity(self, severity: str) -> List[Dict]:
        """
        Get alerts filtered by severity
        
        Args:
            severity: Severity level to filter
            
        Returns:
            List of matching alerts
        """
        return [
            alert for alert in self.alert_history
            if alert['severity'] == severity
        ]
    
    def clear_history(self) -> None:
        """Clear alert history"""
        self.alert_history = []
        logger.info("Alert history cleared")
    
    def get_statistics(self) -> Dict:
        """
        Get alert statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_alerts': self.total_alerts,
            'alerts_by_type': self.alerts_by_type.copy(),
            'history_size': len(self.alert_history),
            'audio_enabled': self.enable_audio,
            'visual_enabled': self.enable_visual
        }
    
    def send_email_alert(
        self,
        violation: Dict,
        recipients: List[str]
    ) -> bool:
        """
        Send email alert (placeholder for future implementation)
        
        Args:
            violation: Violation dictionary
            recipients: List of email addresses
            
        Returns:
            True if successful
        """
        logger.info(f"Email alert would be sent to: {recipients}")
        # TODO: Implement email sending using SMTP
        return False
    
    def send_sms_alert(
        self,
        violation: Dict,
        phone_numbers: List[str]
    ) -> bool:
        """
        Send SMS alert (placeholder for future implementation)
        
        Args:
            violation: Violation dictionary
            phone_numbers: List of phone numbers
            
        Returns:
            True if successful
        """
        logger.info(f"SMS alert would be sent to: {phone_numbers}")
        # TODO: Implement SMS sending using Twilio or similar service
        return False
    
    def send_webhook(
        self,
        violation: Dict,
        webhook_url: str
    ) -> bool:
        """
        Send webhook notification (placeholder for future implementation)
        
        Args:
            violation: Violation dictionary
            webhook_url: Webhook URL
            
        Returns:
            True if successful
        """
        logger.info(f"Webhook would be sent to: {webhook_url}")
        # TODO: Implement webhook POST request
        return False