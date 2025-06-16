from enum import Enum

class   RegionType(str, Enum):
    """Document region type enumeration"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    TITLE = "title"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CAPTION = "caption"
    SIDEBAR = "sidebar"
    OTHER = "other"

class ContentType(str, Enum):
    """Content type enumeration"""
    TEXT_PARAGRAPH = "text_paragraph"
    TEXT_TITLE = "text_title"
    TEXT_HEADER = "text_header"
    TEXT_FOOTER = "text_footer"
    TEXT_LIST = "text_list"
    TEXT_CAPTION = "text_caption"
    TABLE_STRUCTURED = "table_structured"
    IMAGE_REFERENCE = "image_reference"

class EntityType(str, Enum):
    """Entity type enumeration"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    PRODUCT = "product"
    CONCEPT = "concept"
    NUMERIC = "numeric"
    EVENT = "event"
    TECHNOLOGY = "technology"
    DOCUMENT = "document"
    OTHER = "other"

class RelationType(str, Enum):
    """Relation type enumeration"""
    IS_A = "is_a"                           # Subclass relationship
    PART_OF = "part_of"                     # Part-whole relationship
    LOCATED_IN = "located_in"               # Spatial relationship
    WORKS_FOR = "works_for"                 # Employment relationship
    CREATED_BY = "created_by"               # Creation relationship
    CONTAINS = "contains"                   # Containment relationship
    RELATED_TO = "related_to"               # General association
    TEMPORAL = "temporal"                   # Time-based relationship
    CAUSAL = "causal"                       # Cause-effect relationship
    MEMBER_OF = "member_of"                 # Membership relationship
    INSTANCE_OF = "instance_of"             # Instance relationship
    SIMILAR_TO = "similar_to"               # Similarity relationship
    DEPENDS_ON = "depends_on"               # Dependency relationship
    USED_BY = "used_by"                     # Usage relationship
    OWNS = "owns"                           # Ownership relationship
    AFFECTS = "affects"                     # Influence relationship
    PRECEDES = "precedes"                   # Temporal precedence
    FOLLOWS = "follows"                     # Temporal succession
    COMPOSED_OF = "composed_of"             # Composition relationship
    PRODUCES = "produces"                   # Production relationship
    OTHER = "other"

class CellType(str, Enum):
    """Table cell type enumeration"""
    HEADER = "header"
    DATA = "data"
    FOOTER = "footer"
    MERGED = "merged"
    EMPTY = "empty"

class PageOrientation(str, Enum):
    """Page orientation enumeration"""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"

class ExtractionMethod(str, Enum):
    """Extraction method enumeration"""
    GPT4V_VISUAL = "gpt4v_visual"
    TABLE_TRANSFORMER = "table_transformer"
    OCR_TESSERACT = "ocr_tesseract"
    HYBRID = "hybrid"
    MANUAL = "manual"

class ConfidenceLevel(str, Enum):
    """Confidence level enumeration"""
    VERY_HIGH = "very_high"     # 0.9-1.0
    HIGH = "high"               # 0.8-0.9
    MEDIUM = "medium"           # 0.6-0.8
    LOW = "low"                 # 0.4-0.6
    VERY_LOW = "very_low"       # 0.0-0.4