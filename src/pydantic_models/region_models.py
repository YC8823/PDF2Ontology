from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from .enums import RegionType, PageOrientation

class BoundingBox(BaseModel):
    """Bounding box coordinates model"""
    x: float = Field(
        description="Left coordinate (relative to image width, 0-1)", 
        ge=0, le=1
    )
    y: float = Field(
        description="Top coordinate (relative to image height, 0-1)", 
        ge=0, le=1
    )
    width: float = Field(
        description="Width (relative to image width, 0-1)", 
        ge=0, le=1
    )
    height: float = Field(
        description="Height (relative to image height, 0-1)", 
        ge=0, le=1
    )
    
    @property
    def right(self) -> float:
        """Right edge coordinate"""
        return self.x + self.width
    
    @property
    def bottom(self) -> float:
        """Bottom edge coordinate"""
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        """Center X coordinate"""
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Center Y coordinate"""
        return self.y + self.height / 2
    
    @property
    def area(self) -> float:
        """Bounding box area"""
        return self.width * self.height

class DocumentRegion(BaseModel):
    """Document region model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: RegionType = Field(description="Region type")
    bbox: BoundingBox = Field(description="Bounding box coordinates")
    confidence: float = Field(
        description="Detection confidence (0-1)", 
        ge=0, le=1
    )
    content_description: str = Field(description="Content description")
    page_number: int = Field(description="Page number", ge=1)
    reading_order: Optional[int] = Field(
        description="Reading order within page", 
        default=None
    )
    language: Optional[str] = Field(
        description="Detected language (ISO 639-1)", 
        default=None
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class DocumentLayout(BaseModel):
    """Document layout analysis result"""
    regions: List[DocumentRegion] = Field(description="Detected regions")
    page_format: str = Field(description="Page format (e.g., A4, Letter)")
    orientation: PageOrientation = Field(description="Page orientation")
    total_regions: int = Field(description="Total number of regions")
    page_number: int = Field(description="Page number", ge=1)
    image_dimensions: Optional[Dict[str, int]] = Field(
        description="Image dimensions (width, height)", 
        default=None
    )
    processing_metadata: Dict[str, Any] = Field(
        description="Processing metadata", 
        default_factory=dict
    )
    analysis_summary: str = Field(description="Analysis summary")