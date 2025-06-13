"""
Pydantic models for structured data representation throughout the pipeline
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class RegionType(str, Enum):
    """Types of regions that can be detected in documents"""
    TEXT = "text"
    TABLE = "table" 
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    TITLE = "title"
    CAPTION = "caption"
    SIDEBAR = "sidebar"

class BoundingBox(BaseModel):
    """Bounding box coordinates with validation"""
    x: int = Field(..., ge=0, description="Left x coordinate")
    y: int = Field(..., ge=0, description="Top y coordinate") 
    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")
    
    @field_validator('width', 'height')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Width and height must be positive')
        return v
    
    def to_corners(self) -> tuple:
        """Convert to (x1, y1, x2, y2) format"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def area(self) -> int:
        """Calculate area of bounding box"""
        return self.width * self.height
    
    def center(self) -> tuple:
        """Get center point of bounding box"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    model_config = {
        "extra": "forbid",  
        "json_schema_extra": {
            "example": {
                "x": 50,
                "y": 100, 
                "width": 400,
                "height": 200
            }
        }
    }

class RegionMetadata(BaseModel):
    """Specific metadata fields for regions - OpenAI compatible"""
    font_size: Optional[float] = Field(None, description="Estimated font size if applicable")
    text_alignment: Optional[str] = Field(None, description="Text alignment (left, center, right)")
    background_color: Optional[str] = Field(None, description="Background color if detected")
    border_detected: Optional[bool] = Field(None, description="Whether region has visible borders")
    column_index: Optional[int] = Field(None, description="Column index in multi-column layout")

class DocumentRegion(BaseModel):
    """Represents a detected region in the document"""
    region_id: str = Field(..., description="Unique identifier for the region")
    region_type: RegionType = Field(..., description="Type of the detected region")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for detection") # confidence 的必要性？哪里来的？
    content_description: str = Field(..., description="Brief description of region content")
    page_number: int = Field(..., ge=0, description="Page number (0-indexed)")
    reading_order: Optional[int] = Field(None, description="Reading order sequence") # 有没有用？
    #metadata: Optional[RegionMetadata] = Field(None, description="Additional structured metadata for this region") # metadata examples

    model_config = {
        "extra": "forbid",  
        "json_schema_extra": {
            "example": {
                "region_id": "region_1",
                "region_type": "text",
                "bbox": {"x": 50, "y": 100, "width": 400, "height": 200},
                "confidence": 0.95,
                "content_description": "Main body paragraph discussing quarterly financial results",
                "page_number": 0,
                "reading_order": 2, # 建议的reading order 还是他读的order？ 
                #"metadata": {}
            }
        }
    }

class ProcessingMetadata(BaseModel):
    """Specific processing metadata - OpenAI compatible"""
    analysis_method: str = Field(..., description="Method used for analysis") #保留
    document_type: Optional[str] = Field(None, description="Inferred document type")
    reading_flow: Optional[str] = Field(None, description="Reading flow pattern") #保留
    complexity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Layout complexity score") #可以看看复杂度其他的参数之间的关系
    processing_time_seconds: Optional[float] = Field(None, description="Processing time in seconds") #保留
    # model_version: Optional[str] = Field(None, description="Model version used") 

class ImageDimensions(BaseModel):
    """Image dimensions - OpenAI compatible"""
    width: int = Field(..., description="Image width in pixels", gt=0)
    height: int = Field(..., description="Image height in pixels", gt=0)

class DocumentAnalysisResult(BaseModel):
    """Complete analysis result - OpenAI structured output compatible"""
    document_id: str = Field(..., description="Unique document identifier")
    regions: List[DocumentRegion] = Field(..., description="List of all detected regions in reading order")
    page_layout: str = Field(..., description="Overall description of page layout and structure") #做成后台。
    total_pages: int = Field(..., ge=1, description="Total number of pages analyzed")
    image_dimensions: ImageDimensions = Field(..., description="Original image dimensions")
    # Fixed: Use specific metadata model instead of Dict[str, Any]
    processing_metadata: ProcessingMetadata = Field(..., description="Processing information and statistics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")

    model_config = { 
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "document_id": "doc_20241203_142530",
                "regions": [
                    {
                        "region_id": "region_1",
                        "region_type": "header", 
                        "bbox": {"x": 50, "y": 30, "width": 700, "height": 60},
                        "confidence": 0.98,
                        "content_description": "Document title and header information",
                        "page_number": 0,
                        "reading_order": 1,
                        "metadata": {}
                    }
                ],
                "page_layout": "Single-column layout with header, main content area, and sidebar",
                "total_pages": 1,
                "image_dimensions": {"width": 800, "height": 600},
                "processing_metadata": {
                    "analysis_method": "structured_output",
                    "document_type": "business_report"
                }
            }
        }
    }


class ProcessingStep(BaseModel):
    """Individual step in the processing pipeline"""
    step_name: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class DocumentProcessingState(BaseModel):
    """Tracks the state of document processing through the pipeline"""
    document_id: str
    input_path: str
    output_dir: str
    current_step: str
    steps: List[ProcessingStep] = Field(default_factory=list)
    final_result: Optional[DocumentAnalysisResult] = None

# Additional specialized models for specific LLM tasks

class TableCell(BaseModel):
    """Individual table cell for structured output"""
    row: int = Field(..., description="Row index (0-based)", ge=0)
    column: int = Field(..., description="Column index (0-based)", ge=0)
    content: str = Field(..., description="Text content of the cell")
    is_header: bool = Field(False, description="True if this cell is a header cell")
    rowspan: int = Field(1, description="Number of rows this cell spans", ge=1)
    colspan: int = Field(1, description="Number of columns this cell spans", ge=1)

class TableStructure(BaseModel):
    """Table structure analysis result for LLM structured output"""
    table_id: str = Field(..., description="Unique identifier for this table")
    bbox: BoundingBox = Field(..., description="Table bounding box coordinates")
    rows: int = Field(..., description="Total number of rows", gt=0)
    columns: int = Field(..., description="Total number of columns", gt=0)
    has_headers: bool = Field(..., description="Whether table has header row(s) or column(s)")
    cells: List[TableCell] = Field(..., description="All table cells with their content and position")
    table_caption: Optional[str] = Field(None, description="Table caption or title if present")
    confidence: float = Field(..., description="Confidence in table structure detection", ge=0.0, le=1.0)
    
    model_config = {  # Fixed for Pydantic V2
        "json_schema_extra": {
            "example": {
                "table_id": "table_1",
                "bbox": {"x": 50, "y": 200, "width": 400, "height": 150},
                "rows": 4,
                "columns": 3,
                "has_headers": True,
                "cells": [
                    {
                        "row": 0,
                        "column": 0,
                        "content": "Quarter",
                        "is_header": True,
                        "rowspan": 1,
                        "colspan": 1
                    }
                ],
                "table_caption": "Quarterly Financial Results",
                "confidence": 0.92
            }
        }
    }

# class ExtractedText(BaseModel):
#     """Extracted text content from a region"""
#     region_id: str = Field(..., description="Reference to the source region ID")
#     extracted_text: str = Field(..., description="Complete extracted text content")
#     language: str = Field("en", description="Detected language code (ISO 639-1)")
#     text_quality: float = Field(..., description="Text extraction quality assessment", ge=0.0, le=1.0)
#     word_count: int = Field(..., description="Number of words in extracted text", ge=0)
#     formatting_preserved: bool = Field(..., description="Whether original text formatting was preserved")

# class ContentExtractionResult(BaseModel):
#     """Complete content extraction result"""
#     text_regions: List[ExtractedText] = Field(..., description="Extracted text from all text regions")
#     total_text_length: int = Field(..., description="Total character count across all regions", ge=0)
#     total_word_count: int = Field(..., description="Total word count across all regions", ge=0)
#     primary_language: str = Field("en", description="Primary language detected in document")
#     extraction_timestamp: datetime = Field(default_factory=datetime.now, description="When extraction was performed")
