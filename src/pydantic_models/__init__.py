"""
Pydantic Models for PDF2Ontology
"""

from .enums import (
    RegionType, ContentType, EntityType, RelationType, 
    CellType, PageOrientation, ExtractionMethod, ConfidenceLevel
)

from .region_models import (
    BoundingBox, DocumentRegion, DocumentLayout
)

from .content_models import (
    FontInfo, TextBlock, CrossPageContent, DocumentContent
)

from .table_models import (
    TableCell, TableHeader, TableStructure, TableData
)

from .knowledge_models import (
    EntityProperty, Entity, Relation, Triplet, 
    OntologyClass, KnowledgeGraph
)

from .pipeline_models import (
    ProcessingStage, ProcessingStatus, StageResult, PipelineResult
)

__all__ = [
    # Enums
    "RegionType", "ContentType", "EntityType", "RelationType",
    "CellType", "PageOrientation", "ExtractionMethod", "ConfidenceLevel",
    
    # Region Models
    "BoundingBox", "DocumentRegion", "DocumentLayout",
    
    # Content Models
    "FontInfo", "TextBlock", "CrossPageContent", "DocumentContent",
    
    # Table Models
    "TableCell", "TableHeader", "TableStructure", "TableData",
    
    # Knowledge Models
    "EntityProperty", "Entity", "Relation", "Triplet", 
    "OntologyClass", "KnowledgeGraph",
    
    # Pipeline Models
    "ProcessingStage", "ProcessingStatus", "StageResult", "PipelineResult"
]

# Model validation utilities
def validate_confidence(value: float) -> bool:
    """Validate confidence value is between 0 and 1"""
    return 0.0 <= value <= 1.0

def validate_page_number(value: int) -> bool:
    """Validate page number is positive"""
    return value >= 1

def validate_bbox_coordinates(bbox: BoundingBox) -> bool:
    """Validate bounding box coordinates are within valid range"""
    return (
        0.0 <= bbox.x <= 1.0 and
        0.0 <= bbox.y <= 1.0 and
        0.0 <= bbox.width <= 1.0 and
        0.0 <= bbox.height <= 1.0 and
        bbox.x + bbox.width <= 1.0 and
        bbox.y + bbox.height <= 1.0
    )

# Model creation helpers
def create_text_block(content: str, content_type: ContentType, 
                     page_number: int, confidence: float = 0.8) -> TextBlock:
    """Helper function to create a text block"""
    return TextBlock(
        content=content,
        content_type=content_type,
        page_number=page_number,
        order_in_page=0,
        confidence=confidence,
        word_count=len(content.split()),
        character_count=len(content)
    )

def create_entity(name: str, entity_type: EntityType, 
                 source_text: str, page_number: int, 
                 confidence: float = 0.8) -> Entity:
    """Helper function to create an entity"""
    return Entity(
        name=name,
        entity_type=entity_type,
        confidence=confidence,
        source_text=source_text,
        page_number=page_number
    )

def create_triplet(subject: Entity, predicate: RelationType, 
                  obj: Entity, source_sentence: str, 
                  page_number: int, confidence: float = 0.8) -> Triplet:
    """Helper function to create a triplet"""
    return Triplet(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=confidence,
        source_sentence=source_sentence,
        page_number=page_number
    )