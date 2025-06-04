"""
Input validation and error handling utilities
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from .image_utils import ImageProcessor

logger = logging.getLogger(__name__)

class DocumentValidator:
    """Validates document inputs and processing parameters"""
    
    @staticmethod
    def validate_input_file(file_path: str) -> Dict[str, Any]:
        """Validate input document file"""
        
        try:
            # Basic file validation
            if not file_path:
                raise ValueError("File path cannot be empty")
            
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type and validate accordingly
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                return DocumentValidator._validate_pdf(file_path)
            elif file_ext in ImageProcessor.SUPPORTED_FORMATS:
                return ImageProcessor.validate_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"File validation failed: {str(e)}")
            raise
    
    @staticmethod
    def _validate_pdf(pdf_path: str) -> Dict[str, Any]:
        """Validate PDF file"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            if page_count == 0:
                raise ValueError("PDF contains no pages")
            
            if page_count > 50:  # Reasonable limit
                raise ValueError(f"PDF too large: {page_count} pages")
            
            # Get first page dimensions
            page = doc.load_page(0)
            rect = page.rect
            
            doc.close()
            
            return {
                "page_count": page_count,
                "width": int(rect.width),
                "height": int(rect.height),
                "format": "PDF",
                "file_size": os.path.getsize(pdf_path),
                "is_valid": True
            }
            
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {str(e)}")
    
    @staticmethod
    def validate_output_directory(output_dir: str) -> str:
        """Validate and create output directory"""
        
        if not output_dir:
            raise ValueError("Output directory cannot be empty")
        
        output_dir = os.path.abspath(output_dir)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(output_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            return output_dir
            
        except PermissionError:
            raise PermissionError(f"No write permission for directory: {output_dir}")
        except Exception as e:
            raise ValueError(f"Invalid output directory: {str(e)}")
    
    @staticmethod
    def validate_processing_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate processing parameters"""
        
        validated_params = {}
        
        # API key validation
        api_key = params.get('openai_api_key')
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid OpenAI API key required")
        validated_params['openai_api_key'] = api_key
        
        # Model name validation
        model_name = params.get('model_name', 'gpt-4o')
        if model_name not in ['gpt-4o', 'gpt-4-vision-preview']:
            raise ValueError(f"Unsupported model: {model_name}")
        validated_params['model_name'] = model_name
        
        # Temperature validation
        temperature = params.get('temperature', 0)
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        validated_params['temperature'] = temperature
        
        # Max tokens validation
        max_tokens = params.get('max_tokens', 4096)
        if not isinstance(max_tokens, int) or max_tokens < 100 or max_tokens > 8192:
            raise ValueError("Max tokens must be between 100 and 8192")
        validated_params['max_tokens'] = max_tokens
        
        return validated_params