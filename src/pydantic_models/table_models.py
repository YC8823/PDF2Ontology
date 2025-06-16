from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import uuid
from .enums import CellType, ExtractionMethod
from .region_models import BoundingBox

class TableCell(BaseModel):
    """Table cell model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    row: int = Field(description="Row index (0-based)", ge=0)
    col: int = Field(description="Column index (0-based)", ge=0)
    content: str = Field(description="Cell content")
    rowspan: int = Field(description="Number of rows spanned", ge=1, default=1)
    colspan: int = Field(description="Number of columns spanned", ge=1, default=1)
    cell_type: CellType = Field(description="Cell type", default=CellType.DATA)
    confidence: float = Field(
        description="Recognition confidence (0-1)", 
        ge=0, le=1
    )
    data_type: Optional[str] = Field(
        description="Inferred data type", 
        default=None
    )
    formatting: Optional[Dict[str, Any]] = Field(
        description="Cell formatting information", 
        default=None
    )
    bbox: Optional[BoundingBox] = Field(
        description="Cell bounding box", 
        default=None
    )
    is_empty: bool = Field(description="Whether cell is empty", default=False)
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class TableHeader(BaseModel):
    """Table header model"""
    level: int = Field(description="Header level (0=top level)", ge=0)
    columns: List[int] = Field(description="Column indices covered")
    text: str = Field(description="Header text")
    alignment: Optional[str] = Field(description="Text alignment", default=None)

class TableStructure(BaseModel):
    """Table structure model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rows: int = Field(description="Number of rows", ge=1)
    cols: int = Field(description="Number of columns", ge=1)
    cells: List[TableCell] = Field(description="List of table cells")
    headers: List[str] = Field(
        description="Column headers", 
        default_factory=list
    )
    multi_level_headers: List[TableHeader] = Field(
        description="Multi-level headers", 
        default_factory=list
    )
    caption: Optional[str] = Field(description="Table caption", default=None)
    page_number: int = Field(description="Page number", ge=1)
    bbox: Optional[BoundingBox] = Field(
        description="Table bounding box", 
        default=None
    )
    extraction_method: ExtractionMethod = Field(
        description="Extraction method used"
    )
    structure_confidence: float = Field(
        description="Structure recognition confidence (0-1)", 
        ge=0, le=1
    )
    has_merged_cells: bool = Field(
        description="Whether table has merged cells", 
        default=False
    )
    table_type: Optional[str] = Field(
        description="Table type classification", 
        default=None
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class TableData(BaseModel):
    """Table data model"""
    structure: TableStructure = Field(description="Table structure")
    data_rows: List[Dict[str, Any]] = Field(description="Data rows as dictionaries")
    raw_data: Optional[List[List[str]]] = Field(
        description="Raw data as 2D array", 
        default=None
    )
    column_types: Dict[str, str] = Field(
        description="Inferred column data types", 
        default_factory=dict
    )
    statistics: Dict[str, Any] = Field(
        description="Table statistics", 
        default_factory=dict
    )
    quality_metrics: Dict[str, float] = Field(
        description="Data quality metrics", 
        default_factory=dict
    )
    notes: Optional[str] = Field(description="Additional notes", default=None)
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )