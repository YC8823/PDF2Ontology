"""
Pydantic models for semantic table extraction using structured output
File: src/pydantic_models/semantic_table_models.py
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TableTypeEnum(str, Enum):
    """Enumeration of table types"""
    DATA_TABLE = "data_table"
    COMPARISON_TABLE = "comparison_table" 
    SPECIFICATION_TABLE = "specification_table"
    FINANCIAL_TABLE = "financial_table"
    PARAMETER_TABLE = "parameter_table"
    OTHER = "other"


class ComplexityLevelEnum(str, Enum):
    """Enumeration of table complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class DataRelationship(BaseModel):
    """Represents a semantic relationship between row header and column values"""
    row_header: str = Field(description="The row header/label")
    values: Dict[str, str] = Field(
        default_factory=dict, 
        description="Mapping of column headers to cell values. Can be empty for single-column tables."
    )
    single_value: Optional[str] = Field(
        default=None, 
        description="For simple two-column tables where there's just one value per row"
    )
    row_notes: Optional[str] = Field(
        default=None, 
        description="Any notes specific to this row"
    )
    
    class Config:
        """Pydantic configuration to handle missing fields gracefully"""
        extra = "ignore"  # Ignore extra fields that aren't defined
        validate_assignment = True


class SemanticTableStructure(BaseModel):
    """Structure information for a semantically analyzed table"""
    rows: int = Field(ge=1, description="Number of rows including headers")
    columns: int = Field(ge=1, description="Number of columns including headers") 
    has_row_headers: bool = Field(description="Whether table has row headers")
    has_column_headers: bool = Field(description="Whether table has column headers")
    header_levels: int = Field(ge=1, description="Number of header levels (1 for simple, >1 for hierarchical)")


class SemanticTableData(BaseModel):
    """Complete semantic representation of an extracted table"""
    table_id: str = Field(description="Unique identifier for this table")
    title: Optional[str] = Field(default=None, description="Descriptive title or topic of the table")
    table_type: TableTypeEnum = Field(description="Classification of table type")
    
    structure: SemanticTableStructure = Field(description="Table structure information")
    
    headers: Dict[str, List[str]] = Field(
        description="Headers organized by type",
        default_factory=lambda: {"row_headers": [], "column_headers": []}
    )
    
    data_relationships: List[DataRelationship] = Field(
        description="Semantic relationships between headers and values",
        default_factory=list
    )
    
    notes: Optional[str] = Field(default=None, description="Any special observations about this table")
    
    class Config:
        """Pydantic configuration to handle missing fields gracefully"""
        extra = "ignore"
        validate_assignment = True


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process"""
    confidence: float = Field(ge=0.0, le=1.0, description="Overall extraction confidence score")
    detected_language: str = Field(description="Primary language detected (e.g., 'de', 'en', 'mixed')")
    data_types: List[str] = Field(description="Types of data found (e.g., 'numeric', 'text', 'units')")
    complexity_level: ComplexityLevelEnum = Field(description="Assessment of table complexity")


class TableSummary(BaseModel):
    """High-level summary of all tables found"""
    total_tables: int = Field(ge=1, description="Total number of tables detected")
    main_topic: str = Field(description="Brief description of overall table content")
    document_type: Optional[str] = Field(default=None, description="Type of document (e.g., 'technical_specification', 'datasheet')")


class SemanticTableExtraction(BaseModel):
    """Complete result of semantic table extraction using structured output"""
    table_summary: TableSummary = Field(description="High-level summary of extraction")
    
    tables: List[SemanticTableData] = Field(
        description="List of all extracted tables with semantic data",
        min_items=1
    )
    
    extraction_metadata: ExtractionMetadata = Field(description="Metadata about the extraction process")
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        extra = "ignore"  # Ignore extra fields that aren't defined


# Example usage and validation
def example_semantic_extraction():
    """Example of creating a SemanticTableExtraction object"""
    
    # Example data relationship
    relationship = DataRelationship(
        row_header="durchfluss",
        values={
            "min.": "20",
            "max.": "50", 
            "avg.": "35",
            "unit": "l/min"
        },
        row_notes="Flow rate measurements"
    )
    
    # Example table structure
    structure = SemanticTableStructure(
        rows=5,
        columns=4,
        has_row_headers=True,
        has_column_headers=True,
        header_levels=1
    )
    
    # Example table data
    table_data = SemanticTableData(
        table_id="table_1",
        title="Hydraulic Parameters",
        table_type=TableTypeEnum.SPECIFICATION_TABLE,
        structure=structure,
        headers={
            "row_headers": ["durchfluss", "druck", "temperatur"],
            "column_headers": ["min.", "max.", "avg.", "unit"]
        },
        data_relationships=[relationship],
        notes="Technical specifications for hydraulic system"
    )
    
    # Example metadata
    metadata = ExtractionMetadata(
        confidence=0.95,
        detected_language="de",
        data_types=["numeric", "text", "units"],
        complexity_level=ComplexityLevelEnum.MODERATE
    )
    
    # Example summary
    summary = TableSummary(
        total_tables=1,
        main_topic="Hydraulic system specifications",
        document_type="technical_datasheet"
    )
    
    # Complete extraction result
    extraction = SemanticTableExtraction(
        table_summary=summary,
        tables=[table_data],
        extraction_metadata=metadata
    )
    
    return extraction
