from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from .enums import ContentType, ExtractionMethod
from .region_models import BoundingBox

class FontInfo(BaseModel):
    """Font information model"""
    family: Optional[str] = Field(description="Font family", default=None)
    size: Optional[float] = Field(description="Font size", default=None)
    weight: Optional[str] = Field(description="Font weight", default=None)
    style: Optional[str] = Field(description="Font style", default=None)
    color: Optional[str] = Field(description="Font color", default=None)

class TextBlock(BaseModel):
    """Text block model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(description="Text content")
    content_type: ContentType = Field(description="Content type")
    page_number: int = Field(description="Page number", ge=1)
    order_in_page: int = Field(description="Order within page", ge=0)
    confidence: float = Field(
        description="Extraction confidence (0-1)", 
        ge=0, le=1
    )
    bbox: Optional[BoundingBox] = Field(
        description="Bounding box coordinates", 
        default=None
    )
    font_info: Optional[FontInfo] = Field(
        description="Font information", 
        default=None
    )
    language: Optional[str] = Field(
        description="Detected language (ISO 639-1)", 
        default=None
    )
    word_count: Optional[int] = Field(
        description="Number of words", 
        default=None
    )
    character_count: Optional[int] = Field(
        description="Number of characters", 
        default=None
    )
    extraction_method: ExtractionMethod = Field(
        description="Extraction method used",
        default=ExtractionMethod.GPT4V_VISUAL
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class CrossPageContent(BaseModel):
    """Cross-page content model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_blocks: List[TextBlock] = Field(description="List of content blocks")
    merged_content: str = Field(description="Merged complete content")
    start_page: int = Field(description="Starting page number", ge=1)
    end_page: int = Field(description="Ending page number", ge=1)
    content_type: ContentType = Field(description="Content type")
    merge_confidence: float = Field(
        description="Merge confidence (0-1)", 
        ge=0, le=1
    )
    merge_method: str = Field(
        description="Method used for merging", 
        default="semantic_similarity"
    )
    total_word_count: Optional[int] = Field(
        description="Total word count", 
        default=None
    )
    languages: List[str] = Field(
        description="Detected languages", 
        default_factory=list
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class DocumentContent(BaseModel):
    """Document content model"""
    text_blocks: List[TextBlock] = Field(description="List of text blocks")
    cross_page_contents: List[CrossPageContent] = Field(
        description="List of cross-page contents"
    )
    total_pages: int = Field(description="Total number of pages", ge=1)
    total_text_blocks: int = Field(description="Total number of text blocks")
    total_word_count: Optional[int] = Field(
        description="Total word count", 
        default=None
    )
    total_character_count: Optional[int] = Field(
        description="Total character count", 
        default=None
    )
    languages_detected: List[str] = Field(
        description="Detected languages", 
        default_factory=list
    )
    extraction_summary: str = Field(description="Extraction summary")
    processing_time: Optional[float] = Field(
        description="Processing time in seconds", 
        default=None
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )