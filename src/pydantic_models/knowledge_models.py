from typing import List, Dict, Any, Optional, Set, Union
from pydantic import BaseModel, Field
import uuid
from datetime import datetime
from .enums import EntityType, RelationType

class EntityProperty(BaseModel):
    """Entity property model"""
    name: str = Field(description="Property name")
    value: Union[str, int, float, bool] = Field(description="Property value")
    data_type: str = Field(description="Data type of the value")
    confidence: float = Field(
        description="Property confidence (0-1)", 
        ge=0, le=1
    )
    source: Optional[str] = Field(description="Source of the property", default=None)

class Entity(BaseModel):
    """Entity model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Entity name")
    entity_type: EntityType = Field(description="Entity type")
    aliases: List[str] = Field(description="List of aliases", default_factory=list)
    description: Optional[str] = Field(description="Entity description", default=None)
    properties: List[EntityProperty] = Field(
        description="Entity properties", 
        default_factory=list
    )
    confidence: float = Field(
        description="Recognition confidence (0-1)", 
        ge=0, le=1
    )
    source_text: str = Field(description="Source text")
    page_number: int = Field(description="Page number", ge=1)
    mentions: List[str] = Field(
        description="Text mentions of the entity", 
        default_factory=list
    )
    context: Optional[str] = Field(
        description="Surrounding context", 
        default=None
    )
    canonical_form: Optional[str] = Field(
        description="Canonical form of the entity", 
        default=None
    )
    uri: Optional[str] = Field(
        description="Unique resource identifier", 
        default=None
    )
    external_ids: Dict[str, str] = Field(
        description="External system identifiers", 
        default_factory=dict
    )
    created_at: datetime = Field(
        description="Creation timestamp", 
        default_factory=datetime.now
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class Relation(BaseModel):
    """Relation model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = Field(description="Subject entity ID")
    predicate: RelationType = Field(description="Relation type")
    object_id: str = Field(description="Object entity ID")
    confidence: float = Field(
        description="Relation confidence (0-1)", 
        ge=0, le=1
    )
    source_text: str = Field(description="Source text")
    context: Optional[str] = Field(description="Context", default=None)
    page_number: int = Field(description="Page number", ge=1)
    weight: float = Field(
        description="Relation weight/strength", 
        default=1.0, 
        ge=0
    )
    temporal_info: Optional[str] = Field(
        description="Temporal information", 
        default=None
    )
    negated: bool = Field(
        description="Whether the relation is negated", 
        default=False
    )
    modality: Optional[str] = Field(
        description="Modality (e.g., possible, certain)", 
        default=None
    )
    evidence: List[str] = Field(
        description="Supporting evidence", 
        default_factory=list
    )
    created_at: datetime = Field(
        description="Creation timestamp", 
        default_factory=datetime.now
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class Triplet(BaseModel):
    """Knowledge triplet model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: Entity = Field(description="Subject entity")
    predicate: RelationType = Field(description="Predicate")
    object: Entity = Field(description="Object entity")
    confidence: float = Field(
        description="Triplet confidence (0-1)", 
        ge=0, le=1
    )
    source_sentence: str = Field(description="Source sentence")
    page_number: int = Field(description="Page number", ge=1)
    extraction_method: str = Field(
        description="Extraction method", 
        default="llm_extraction"
    )
    validated: bool = Field(
        description="Whether triplet has been validated", 
        default=False
    )
    validation_score: Optional[float] = Field(
        description="Validation score", 
        default=None
    )
    temporal_scope: Optional[str] = Field(
        description="Temporal scope of the relation", 
        default=None
    )
    geographical_scope: Optional[str] = Field(
        description="Geographical scope", 
        default=None
    )
    created_at: datetime = Field(
        description="Creation timestamp", 
        default_factory=datetime.now
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class OntologyClass(BaseModel):
    """Ontology class model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Class name")
    label: str = Field(description="Human-readable label")
    description: Optional[str] = Field(description="Class description", default=None)
    parent_classes: List[str] = Field(
        description="Parent class IDs", 
        default_factory=list
    )
    properties: List[str] = Field(
        description="Associated properties", 
        default_factory=list
    )
    instances: List[str] = Field(
        description="Instance entity IDs", 
        default_factory=list
    )
    namespace: Optional[str] = Field(description="Namespace URI", default=None)
    created_at: datetime = Field(
        description="Creation timestamp", 
        default_factory=datetime.now
    )

class KnowledgeGraph(BaseModel):
    """Knowledge graph model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = Field(description="Knowledge graph name", default=None)
    description: Optional[str] = Field(
        description="Knowledge graph description", 
        default=None
    )
    entities: List[Entity] = Field(description="List of entities")
    relations: List[Relation] = Field(description="List of relations")
    triplets: List[Triplet] = Field(description="List of triplets")
    ontology_classes: List[OntologyClass] = Field(
        description="Ontology classes", 
        default_factory=list
    )
    namespaces: Dict[str, str] = Field(
        description="Namespace mappings", 
        default_factory=dict
    )
    statistics: Dict[str, int] = Field(
        description="Graph statistics", 
        default_factory=dict
    )
    quality_metrics: Dict[str, float] = Field(
        description="Quality metrics", 
        default_factory=dict
    )
    version: str = Field(description="Version identifier", default="1.0")
    created_at: datetime = Field(
        description="Creation timestamp", 
        default_factory=datetime.now
    )
    last_updated: datetime = Field(
        description="Last update timestamp", 
        default_factory=datetime.now
    )
    source_documents: List[str] = Field(
        description="Source document identifiers", 
        default_factory=list
    )
    extraction_config: Dict[str, Any] = Field(
        description="Extraction configuration used", 
        default_factory=dict
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )

class ExtractedEntity(BaseModel):
    """Entity extraction result for structured output"""
    name: str = Field(description="Entity name or identifier")
    entity_type: EntityType = Field(description="Type of entity")
    value: Optional[str] = Field(default=None, description="Entity value if applicable")
    unit: Optional[str] = Field(default=None, description="Unit of measurement if applicable")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    description: Optional[str] = Field(default=None, description="Entity description")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    source_location: str = Field(description="Location in source data (row/column reference)")


class ExtractedRelation(BaseModel):
    """Relation extraction result for structured output"""
    subject_name: str = Field(description="Name of subject entity")
    predicate: RelationType = Field(description="Type of relation")
    object_name: str = Field(description="Name of object entity")
    confidence: float = Field(ge=0.0, le=1.0, description="Relation confidence")
    evidence: str = Field(description="Supporting evidence from source")
    context: Optional[str] = Field(default=None, description="Additional context")


class TripletExtractionOutput(BaseModel):
    """Structured output for triplet extraction"""
    document_type: str = Field(description="Type of document analyzed")
    extraction_summary: str = Field(description="Brief summary of extraction")
    
    entities: List[ExtractedEntity] = Field(
        description="All entities extracted from the data",
        min_items=0
    )
    
    relations: List[ExtractedRelation] = Field(
        description="All relations extracted from the data", 
        min_items=0
    )
    
    quality_assessment: Dict[str, float] = Field(
        description="Quality metrics for the extraction",
        default_factory=dict
    )
    
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Additional notes about the extraction process"
    )
    
    class Config:
        use_enum_values = True
        validate_assignment = True