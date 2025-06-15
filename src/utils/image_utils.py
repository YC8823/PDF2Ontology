"""
Image processing and validation utilities
"""
import fitz #PyMuPDF
import os
import hashlib
from typing import Tuple, Optional,List
from PIL import Image, ImageEnhance
import cv2
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
    def resize_if_needed(image_path: str, max_dimension: int = 4096) -> str:
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

# image proprocessor for ocr engine       
def preprocess_image_for_ocr(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    Applies preprocessing steps to an image to enhance it for table recognition.
    Specifically uses Gamma Correction to improve contrast in industrial documents.

    Args:
        image: The input image as a NumPy array (in BGR format).
        gamma: The gamma value for correction. Values < 1 darken the image,
               values > 1 brighten it. Industrial reports often benefit from
               values slightly less than 1.

    Returns:
        The preprocessed image as a NumPy array.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values.
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")

    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, lookup_table)

def crop_image_by_bbox(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crops an image using a bounding box.

    Args:
        image: The source image as a NumPy array.
        bbox: A list of 4 integers [x1, y1, x2, y2].

    Returns:
        The cropped image region.
    """
    x1, y1, x2, y2 = [int(p) for p in bbox]
    return image[y1:y2, x1:x2]

def convert_pdf_to_images(
    pdf_path: str, 
    dpi: int = 300
) -> List[Image.Image]:
    """
    Converts a PDF file into a list of high-quality, enhanced PIL Images.

    Args:
        pdf_path (str): The file path to the PDF.
        dpi (int): Dots per inch, affects the resolution of the output images. 
                   300 is a good value for OCR.

    Returns:
        List[Image.Image]: A list of PIL Image objects, one for each page.
    """
    print(f"Converting PDF '{pdf_path}' to images at {dpi} DPI...")
    images = []
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # Iterate through each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Render page to a pixmap (a raster image)
            pix = page.get_pixmap(dpi=dpi)
            
            # Convert pixmap to a NumPy array for OpenCV
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Check if image has an alpha channel and handle it
            if img_data.shape[2] == 4:  # RGBA
                # Create a white background
                bg_color = (255, 255, 255)
                alpha = img_data[:, :, 3] / 255.0
                # Blend the image with the white background
                img_bgr = (img_data[:, :, :3] * alpha[:, :, np.newaxis]).astype(np.uint8)
                bg = (np.array(bg_color) * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
                img_bgr = cv2.add(img_bgr, bg)
            else: # RGB
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            # --- Image Enhancement using OpenCV ---
            # 1. Convert to grayscale for better contrast analysis
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 2. Apply a sharpening kernel to enhance text and lines
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
            
            # 3. Convert back to a PIL Image (as required by the processor)
            final_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
            images.append(final_image)
            
        doc.close()
        print(f"Successfully converted {len(images)} pages.")
        return images

    except Exception as e:
        print(f"Error processing PDF file: {e}")
        return []