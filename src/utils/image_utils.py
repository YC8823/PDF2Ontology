"""
Optimized unified image processing utilities 
"""

import cv2
import numpy as np
import os
import hashlib
import logging
import tempfile
import time
from typing import Tuple, Optional, List, Dict, Any, Union
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from langchain_core.runnables import Runnable
from pathlib import Path
import fitz  # PyMuPDF

from ..pydantic_models.document_condition_models import (
    ProcessingAction,
    ImageProcessingResult,
    DocumentCondition
)

logger = logging.getLogger(__name__)


# =============================================================================
# CORE IMAGE PROCESSING CLASS (Unified & Optimized)
# =============================================================================

class ImageProcessor:
    """
    Unified image processor with both basic and advanced capabilities
    Eliminates redundancy while maintaining backward compatibility
    """
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    MAX_DIMENSION = 4096
    MAX_FILE_SIZE_MB = 20
    
    def __init__(self, temp_dir: Optional[str] = None, enable_advanced: bool = True):
        """
        Initialize image processor
        
        Args:
            temp_dir: Temporary directory for intermediate files
            enable_advanced: Whether to enable advanced processing features
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.enable_advanced = enable_advanced
        self.intermediate_files = []
    
    # =============================================================================
    # VALIDATION & UTILITY FUNCTIONS
    # =============================================================================
    
    @classmethod
    def validate_image(cls, image_path: str) -> Dict[str, Any]:
        """Validate image file and return metadata"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        file_size = os.path.getsize(image_path)
        if file_size > cls.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB")
        
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                if width > cls.MAX_DIMENSION or height > cls.MAX_DIMENSION:
                    raise ValueError(f"Image too large: {width}x{height}")
                
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
    
    # =============================================================================
    # UNIFIED PREPROCESSING METHODS
    # =============================================================================
    
    def preprocess_image(self, image_path: str, 
                        processing_level: str = "basic",
                        custom_actions: List[ProcessingAction] = None,
                        **kwargs) -> Union[str, ImageProcessingResult]:
        """
        Unified preprocessing method that replaces multiple redundant functions
        
        Args:
            image_path: Path to input image
            processing_level: "basic", "intermediate", "advanced", or "custom"
            custom_actions: List of specific actions (for custom level)
            **kwargs: Additional parameters (gamma, enhance_quality, etc.)
            
        Returns:
            Processed image path or detailed result object
        """
        
        if processing_level == "basic":
            return self._preprocess_basic(image_path, **kwargs)
        elif processing_level == "intermediate":
            return self._preprocess_intermediate(image_path, **kwargs)
        elif processing_level == "advanced":
            return self._preprocess_advanced(image_path, **kwargs)
        elif processing_level == "custom":
            if not custom_actions:
                raise ValueError("custom_actions required for custom processing level")
            return self._process_with_actions(image_path, custom_actions, **kwargs)
        else:
            raise ValueError(f"Unknown processing level: {processing_level}")
    
    def _preprocess_basic(self, image_path: str, 
                         enhance_quality: bool = True,
                         gamma: float = 0.8) -> str:
        """
        Basic preprocessing (combines original preprocess_image + preprocess_image_for_ocr)
        """
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if enhance_quality:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
            
            if gamma != 1.0:
                img_array = np.array(img)
                corrected = self._apply_gamma_correction(img_array, gamma)
                img = Image.fromarray(corrected)
            
            output_path = self._get_output_path(image_path, "basic")
            img.save(output_path, 'PNG', quality=95)
            
            return output_path
    
    def _preprocess_intermediate(self, image_path: str, **kwargs) -> ImageProcessingResult:
        """Intermediate preprocessing with common enhancements"""
        actions = [
            ProcessingAction.ENHANCE_CONTRAST,
            ProcessingAction.DENOISE,
            ProcessingAction.SHARPEN,
            ProcessingAction.CROP_MARGINS
        ]
        return self._process_with_actions(image_path, actions)
    
    def _preprocess_advanced(self, image_path: str, **kwargs) -> ImageProcessingResult:
        """Advanced preprocessing with comprehensive enhancements"""
        actions = [
            ProcessingAction.DESKEW,
            ProcessingAction.ENHANCE_CONTRAST,
            ProcessingAction.REMOVE_SHADOWS,
            ProcessingAction.DENOISE,
            ProcessingAction.SHARPEN,
            ProcessingAction.ADAPTIVE_THRESHOLD,
            ProcessingAction.CROP_MARGINS
        ]
        return self._process_with_actions(image_path, actions)
    
    def _preprocess_custom(self, image_path: str, 
                          actions: List[ProcessingAction], **kwargs) -> ImageProcessingResult:
        """Custom preprocessing with user-specified actions"""
        return self._process_with_actions(image_path, actions)
    
    # =============================================================================
    # ADVANCED PROCESSING ENGINE
    # =============================================================================
    
    def _process_with_actions(self, image_path: str, 
                             actions: List[ProcessingAction]) -> ImageProcessingResult:
        """Process image with specified actions"""
        if not self.enable_advanced:
            raise ValueError("Advanced processing not enabled")
        
        start_time = time.time()
        self.intermediate_files = []
        
        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            current_image = original_image.copy()
            applied_actions = []
            
            ordered_actions = self._optimize_action_order(actions)
            
            for action in ordered_actions:
                try:
                    processed_image = self._apply_single_action(current_image, action)
                    
                    if len(actions) > 3:
                        intermediate_path = self._save_intermediate_result(processed_image, action)
                        self.intermediate_files.append(intermediate_path)
                    
                    current_image = processed_image
                    applied_actions.append(action)
                    
                except Exception as e:
                    logger.warning(f"Failed to apply {action.value}: {str(e)}")
                    continue
            
            output_path = self._get_output_path(image_path, "advanced")
            cv2.imwrite(output_path, current_image)
            
            quality_improvement = self._estimate_quality_improvement(original_image, current_image)
            processing_time = time.time() - start_time
            
            return ImageProcessingResult(
                input_path=image_path,
                output_path=output_path,
                actions_applied=applied_actions,
                success=True,
                quality_improvement=quality_improvement,
                processing_time=processing_time,
                before_conditions=[],
                after_conditions=[],
                intermediate_files=self.intermediate_files.copy()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Advanced processing failed: {str(e)}")
            
            return ImageProcessingResult(
                input_path=image_path,
                output_path=image_path,
                actions_applied=[],
                success=False,
                processing_time=processing_time,
                before_conditions=[],
                after_conditions=[],
                error_message=str(e),
                intermediate_files=self.intermediate_files.copy()
            )
    
    def _optimize_action_order(self, actions: List[ProcessingAction]) -> List[ProcessingAction]:
        """Optimize the order of processing actions for best results"""
        
        order_groups = {
            # Rotation corrections first (highest priority)
            1: [ProcessingAction.ROTATE_180, ProcessingAction.ROTATE_90_CW, ProcessingAction.ROTATE_90_CCW],
            2: [ProcessingAction.DESKEW, ProcessingAction.STRAIGHTEN_PERSPECTIVE],  # Geometric corrections
            3: [ProcessingAction.CROP_MARGINS],  # Cropping
            4: [ProcessingAction.REMOVE_SHADOWS],  # Shadow removal
            5: [ProcessingAction.ENHANCE_CONTRAST, ProcessingAction.GAMMA_CORRECTION, 
                ProcessingAction.HISTOGRAM_EQUALIZATION],  # Contrast adjustments
            6: [ProcessingAction.DENOISE],  # Noise reduction
            7: [ProcessingAction.SHARPEN],  # Sharpening
            8: [ProcessingAction.BINARIZE, ProcessingAction.ADAPTIVE_THRESHOLD]  # Thresholding last
        }
        
        ordered_actions = []
        for group_order in sorted(order_groups.keys()):
            group_actions = order_groups[group_order]
            for action in actions:
                if action in group_actions and action not in ordered_actions:
                    ordered_actions.append(action)
        
        # Add any remaining actions not in groups
        for action in actions:
            if action not in ordered_actions:
                ordered_actions.append(action)
        
        return ordered_actions
    
    # =============================================================================
    # INDIVIDUAL PROCESSING OPERATIONS (Optimized)
    # =============================================================================
    
    def _apply_single_action(self, image: np.ndarray, action: ProcessingAction) -> np.ndarray:
        """Apply a single processing action"""
        
        action_methods = {
            # Rotation actions
            ProcessingAction.ROTATE_180: lambda img: self._rotate_image(img, 180),
            ProcessingAction.ROTATE_90_CW: lambda img: self._rotate_image(img, 90),
            ProcessingAction.ROTATE_90_CCW: lambda img: self._rotate_image(img, 270),
            
            # Other geometric corrections
            ProcessingAction.DESKEW: self._deskew_image,
            ProcessingAction.STRAIGHTEN_PERSPECTIVE: self._straighten_perspective,
            ProcessingAction.CROP_MARGINS: self._crop_margins_auto,
            
            # Quality enhancements
            ProcessingAction.ENHANCE_CONTRAST: self._enhance_contrast_clahe,
            ProcessingAction.DENOISE: self._denoise_image,
            ProcessingAction.SHARPEN: self._sharpen_image_advanced,
            ProcessingAction.REMOVE_SHADOWS: self._remove_shadows,
            
            # Advanced processing
            ProcessingAction.BINARIZE: self._binarize_image,
            ProcessingAction.ADAPTIVE_THRESHOLD: self._adaptive_threshold,
            ProcessingAction.GAMMA_CORRECTION: self._apply_gamma_correction,
            ProcessingAction.HISTOGRAM_EQUALIZATION: self._histogram_equalization,
        }
        
        if action in action_methods:
            return action_methods[action](image)
        else:
            logger.warning(f"Unknown or unsupported action in ImageProcessor: {action.value}")
            return image
    
    def _enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """Enhanced contrast using CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            enhanced = cv2.merge([l_channel, a_channel, b_channel])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Unified gamma correction"""
        inv_gamma = 1.0 / gamma
        lookup_table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)
        ]).astype("uint8")
        return cv2.LUT(image, lookup_table)
    
    def _sharpen_image_advanced(self, image: np.ndarray) -> np.ndarray:
        """Advanced sharpening"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        alpha = 0.7
        return cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct skew using Hough line detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:
                    angles.append(angle)
            
            if angles:
                skew_angle = np.median(angles)
                height, width = gray.shape
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                new_width = int((height * sin_angle) + (width * cos_angle))
                new_height = int((height * cos_angle) + (width * sin_angle))
                
                rotation_matrix[0, 2] += (new_width / 2) - center[0]
                rotation_matrix[1, 2] += (new_height / 2) - center[1]
                
                return cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle"""
        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        
        if angle in rotation_map:
            return cv2.rotate(image, rotation_map[angle])
        else:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using Non-local Means Denoising"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary using Otsu's method"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if len(image.shape) == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary
    
    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadow artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        background = cv2.GaussianBlur(opened, (21, 21), 0)
        normalized = cv2.divide(gray, background, scale=255)
        
        if len(image.shape) == 3:
            result = image.copy()
            for i in range(3):
                channel = image[:, :, i]
                bg_channel = cv2.GaussianBlur(
                    cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel), (21, 21), 0
                )
                result[:, :, i] = cv2.divide(channel, bg_channel, scale=255)
            return result
        else:
            return normalized
    
    def _crop_margins_auto(self, image: np.ndarray) -> np.ndarray:
        """Automatically crop document margins"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            
            return image[y:y+h, x:x+w]
        return image
    
    def _straighten_perspective(self, image: np.ndarray) -> np.ndarray:
        """Perspective correction (simplified implementation)"""
        return image
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        if len(image.shape) == 3:
            adaptive = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        return adaptive
    
    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization"""
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def resize_if_needed(self, image_path: str, max_dimension: int = 4096) -> str:
        """Resize image if needed"""
        with Image.open(image_path) as img:
            width, height = img.size
            
            if width <= max_dimension and height <= max_dimension:
                return image_path
            
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            output_path = self._get_output_path(image_path, "resized")
            resized_img.save(output_path, 'PNG', quality=95)
            
            return output_path
    
    def crop_by_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crop image by bounding box"""
        x1, y1, x2, y2 = [int(p) for p in bbox]
        return image[y1:y2, x1:x2]
    
    def _get_output_path(self, input_path: str, suffix: str) -> str:
        """Generate output path for processed image"""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = int(time.time() * 1000)
        filename = f"{base_name}_{suffix}_{timestamp}.png"
        return os.path.join(self.temp_dir, filename)
    
    def _save_intermediate_result(self, image: np.ndarray, action: ProcessingAction) -> str:
        """Save intermediate processing result"""
        timestamp = int(time.time() * 1000)
        filename = f"intermediate_{action.value}_{timestamp}.png"
        filepath = os.path.join(self.temp_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath
    
    def _estimate_quality_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Estimate quality improvement between images"""
        try:
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original
                proc_gray = processed
            
            if orig_gray.shape != proc_gray.shape:
                proc_gray = cv2.resize(proc_gray, (orig_gray.shape[1], orig_gray.shape[0]))
            
            orig_contrast = np.std(orig_gray)
            proc_contrast = np.std(proc_gray)
            contrast_improvement = (proc_contrast - orig_contrast) / max(orig_contrast, 1)
            
            orig_edges = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            proc_edges = cv2.Laplacian(proc_gray, cv2.CV_64F).var()
            sharpness_improvement = (proc_edges - orig_edges) / max(orig_edges, 1)
            
            improvement = 0.6 * contrast_improvement + 0.4 * sharpness_improvement
            return max(0.0, min(1.0, improvement + 0.5))
            
        except Exception as e:
            logger.warning(f"Quality estimation failed: {str(e)}")
            return 0.5
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files"""
        for filepath in self.intermediate_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Failed to cleanup {filepath}: {str(e)}")
        self.intermediate_files.clear()


# =============================================================================
# PDF PROCESSING (Enhanced)
# =============================================================================

def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """Enhanced PDF to image conversion with better quality"""
    logger.info(f"Converting PDF '{pdf_path}' to images at {dpi} DPI...")
    images = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if img_data.shape[2] == 4:
                bg_color = (255, 255, 255)
                alpha = img_data[:, :, 3] / 255.0
                img_bgr = (img_data[:, :, :3] * alpha[:, :, np.newaxis]).astype(np.uint8)
                bg = (np.array(bg_color) * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
                img_bgr = cv2.add(img_bgr, bg)
            else:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
            
            final_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
            images.append(final_image)
        
        doc.close()
        logger.info(f"Successfully converted {len(images)} pages.")
        return images
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return []


# =============================================================================
# LANGCHAIN INTEGRATION
# =============================================================================

class RunnableImageProcessor(Runnable):
    """LangChain Runnable wrapper for ImageProcessor"""
    
    def __init__(self, processor: ImageProcessor):
        self.processor = processor
    
    def invoke(self, input_data: dict, config=None) -> dict:
        """LangChain Runnable invoke method"""
        image_path = input_data.get('image_path')
        processing_level = input_data.get('processing_level', 'basic')
        custom_actions = input_data.get('custom_actions')
        
        if not image_path:
            raise ValueError("Input must contain 'image_path' key")
        
        result = self.processor.preprocess_image(
            image_path=image_path,
            processing_level=processing_level,
            custom_actions=custom_actions
        )
        
        return {
            'image_path': image_path,
            'processed_image_path': result.output_path if hasattr(result, 'output_path') else result,
            'processing_result': result,
            'success': result.success if hasattr(result, 'success') else True
        }

