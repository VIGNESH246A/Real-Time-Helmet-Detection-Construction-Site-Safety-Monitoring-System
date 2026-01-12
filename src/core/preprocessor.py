"""
Image preprocessing module for enhancement and normalization
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing for robust detection in various conditions
    """
    
    def __init__(
        self,
        enable_low_light: bool = True,
        enable_noise_reduction: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8
    ):
        """
        Initialize preprocessor
        
        Args:
            enable_low_light: Enable CLAHE for low-light enhancement
            enable_noise_reduction: Enable noise reduction
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_size: CLAHE tile grid size
        """
        self.enable_low_light = enable_low_light
        self.enable_noise_reduction = enable_noise_reduction
        
        # CLAHE for low-light enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
        
        logger.info("Image preprocessor initialized")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed image
        """
        if image is None or image.size == 0:
            logger.warning("Empty image received")
            return image
        
        processed = image.copy()
        
        # Apply enhancements
        if self.enable_low_light:
            processed = self.enhance_low_light(processed)
        
        if self.enable_noise_reduction:
            processed = self.reduce_noise(processed)
        
        return processed
    
    def enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for low-light conditions using CLAHE
        
        Args:
            image: Input BGR image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l_clahe = self.clahe.apply(l)
            
            # Merge channels
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in low-light enhancement: {e}")
            return image
    
    def reduce_noise(
        self,
        image: np.ndarray,
        strength: int = 7,
        search_window: int = 21
    ) -> np.ndarray:
        """
        Reduce noise using Non-local Means Denoising
        
        Args:
            image: Input BGR image
            strength: Filter strength (higher = more denoising)
            search_window: Search window size
            
        Returns:
            Denoised image
        """
        try:
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=strength,
                hColor=strength,
                templateWindowSize=7,
                searchWindowSize=search_window
            )
            return denoised
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return image
    
    def adjust_brightness_contrast(
        self,
        image: np.ndarray,
        brightness: float = 0,
        contrast: float = 1.0
    ) -> np.ndarray:
        """
        Adjust brightness and contrast
        
        Args:
            image: Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast adjustment (0.5 to 3.0)
            
        Returns:
            Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
    
    def sharpen(self, image: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """
        Sharpen image using unsharp mask
        
        Args:
            image: Input image
            amount: Sharpening amount (0 to 2)
            
        Returns:
            Sharpened image
        """
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            
            # Calculate sharpened image
            sharpened = cv2.addWeighted(
                image, 1 + amount,
                blurred, -amount,
                0
            )
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Error in sharpening: {e}")
            return image
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def auto_adjust(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically adjust image exposure and contrast
        
        Args:
            image: Input image
            
        Returns:
            Auto-adjusted image
        """
        try:
            # Convert to YUV
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            # Equalize histogram on Y channel
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to BGR
            adjusted = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Error in auto adjustment: {e}")
            return image
    
    def resize_for_model(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Resize image for model input while maintaining aspect ratio
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Resized image and scale factors (scale_x, scale_y)
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place on canvas
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return canvas, (scale, scale)
    
    def detect_blur(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """
        Detect if image is blurry using Laplacian variance
        
        Args:
            image: Input image
            threshold: Blur threshold (lower = more blurry)
            
        Returns:
            True if blurry, False otherwise
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold