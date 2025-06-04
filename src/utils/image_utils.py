"""
Image processing and validation utilities
"""

import os
import hashlib
from typing import Tuple, Optional
from PIL import Image, ImageEnhance
# import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

class ImageProcessor:
    """Handles image preprocessing, validation, and basic operations"""
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    MAX_DIMENSION = 4096
    MAX_FILE_SIZE_MB = 20
    
    @classmethod
    def validate_image(cls, image_path: str) -> Dict[str, Any]:
        """Validate image file and return metadata"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size > cls.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB")
        
        # Check file extension
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        # Validate image content
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                if width > cls.MAX_DIMENSION or height > cls.MAX_DIMENSION:
                    raise ValueError(f"Image too large: {width}x{height}")
                
                # Generate image hash for caching
                img_hash = cls._generate_image_hash(image_path)
                
                return {
                    "width": width,
                    "height": height,
                    "format": img.format,
                    "mode": img.mode,
                    "file_size": file_size,
                    "image_hash": img_hash,
                    "is_valid": True
                }
                
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
    
    @staticmethod
    def _generate_image_hash(image_path: str) -> str:
        """Generate hash of image content for caching"""
        with open(image_path, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()
    
    @staticmethod
    def preprocess_image(image_path: str, enhance_quality: bool = True) -> str:
        """Preprocess image for better analysis results"""
        
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if enhance_quality:
                # Enhance contrast and sharpness
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
            
            # Save preprocessed image
            output_path = image_path.replace('.', '_processed.')
            img.save(output_path, 'PNG', quality=95)
            
            return output_path
    
    @staticmethod
    def resize_if_needed(image_path: str, max_dimension: int = 2048) -> str:
        """Resize image if it exceeds maximum dimensions"""
        
        with Image.open(image_path) as img:
            width, height = img.size
            
            if width <= max_dimension and height <= max_dimension:
                return image_path
            
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save resized image
            output_path = image_path.replace('.', '_resized.')
            resized_img.save(output_path, 'PNG', quality=95)
            
            return output_path