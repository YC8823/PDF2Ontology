"""
Pydantic models for document condition assessment and image preprocessing
File: src/pydantic_models/document_condition_models.py
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class DocumentCondition(str, Enum):
    """Enhanced document condition enumeration with clear rotation definitions"""
    
    # Geometric orientation issues - clearly defined rotation angles
    SKEWED = "skewed"                    # Small angle tilt (1-15°), content right-side up
    ROTATED_90_CW = "rotated_90_cw"      # Document rotated 90° clockwise, needs 90° CCW to fix
    ROTATED_90_CCW = "rotated_90_ccw"    # Document rotated 90° counter-clockwise, needs 90° CW to fix  
    ROTATED_180 = "rotated_180"          # Document upside down, needs 180° rotation to fix
    
    # Quality issues
    POOR_QUALITY = "poor_quality"
    BLURRY = "blurry"
    LOW_CONTRAST = "low_contrast"
    DARK_BACKGROUND = "dark_background"
    FADED_TEXT = "faded_text"
    
    # Scanning artifacts
    SHADOW_DISTORTION = "shadow_distortion"
    TORN_EDGES = "torn_edges"
    WATERMARKED = "watermarked"
    SCANNED_FROM_COPY = "scanned_from_copy"
    
    # Content characteristics
    HANDWRITTEN = "handwritten"
    MIXED_HANDWRITTEN_TYPED = "mixed_handwritten_typed"
    MULTI_COLUMN = "multi_column"
    COMPLEX_LAYOUT = "complex_layout"


class ProcessingAction(str, Enum):
    """Image processing action enumeration with clear rotation actions"""
    NO_ACTION = "no_action"
    
    # Geometric corrections - clear rotation definitions
    DESKEW = "deskew"                    # Correct small angle skew
    ROTATE_90_CW = "rotate_90_cw"        # Rotate 90° clockwise
    ROTATE_90_CCW = "rotate_90_ccw"      # Rotate 90° counter-clockwise  
    ROTATE_180 = "rotate_180"            # Rotate 180° (fix upside down)
    
    # Quality enhancements
    ENHANCE_CONTRAST = "enhance_contrast"
    DENOISE = "denoise"
    SHARPEN = "sharpen"
    BINARIZE = "binarize"
    REMOVE_SHADOWS = "remove_shadows"
    CROP_MARGINS = "crop_margins"
    STRAIGHTEN_PERSPECTIVE = "straighten_perspective"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    GAMMA_CORRECTION = "gamma_correction"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"


class SeverityLevel(str, Enum):
    """Severity level enumeration"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ConditionDetection(BaseModel):
    """Enhanced condition detection with orientation-specific evidence"""

    condition: DocumentCondition = Field(description="Detected condition")
    severity: SeverityLevel = Field(description="Severity level")
    confidence: float = Field(description="Detection confidence (0-1)", ge=0, le=1)
    evidence: str = Field(description="Specific evidence observed")
    
    # Enhanced fields for better orientation analysis
    orientation_indicators: Optional[List[str]] = Field(
        default_factory=list,
        description="Specific visual indicators for orientation issues"
    )
    alternative_orientations: Optional[List[str]] = Field(
        default_factory=list, 
        description="Other possible orientations considered"
    )
    detection_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning process for this detection"
    )
    
    affected_regions: Optional[List[str]] = Field(
        default_factory=list,
        description="Regions affected by this condition"
    )
    recommended_actions: List[ProcessingAction] = Field(
        description="Recommended processing actions"
    )


class DocumentConditionAssessment(BaseModel):
    """Complete document condition assessment result"""
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    overall_quality: str = Field(description="Overall document quality assessment")
    primary_issues: List[ConditionDetection] = Field(
        description="Primary issues detected in the document"
    )
    secondary_issues: List[ConditionDetection] = Field(
        description="Secondary issues detected",
        default_factory=list
    )
    processing_priority: int = Field(
        description="Processing priority (1=highest, 5=lowest)",
        ge=1, le=5
    )
    estimated_success_rate: float = Field(
        description="Estimated success rate for OCR/analysis (0-1)",
        ge=0, le=1
    )
    processing_recommendations: List[ProcessingAction] = Field(
        description="Ordered list of recommended processing steps"
    )
    special_handling_notes: Optional[str] = Field(
        default=None,
        description="Special notes for handling this document"
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata",
        default_factory=dict
    )
    
    class Config:
        use_enum_values = True


class ImageProcessingResult(BaseModel):
    """Result of image processing operation"""
    processing_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_path: str = Field(description="Input image path")
    output_path: str = Field(description="Output image path")
    actions_applied: List[ProcessingAction] = Field(
        description="Processing actions that were applied"
    )
    success: bool = Field(description="Whether processing was successful")
    quality_improvement: Optional[float] = Field(
        default=None,
        description="Estimated quality improvement (0-1)"
    )
    processing_time: float = Field(description="Processing time in seconds")
    before_conditions: List[DocumentCondition] = Field(
        description="Conditions detected before processing"
    )
    after_conditions: List[DocumentCondition] = Field(
        description="Conditions detected after processing"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    intermediate_files: List[str] = Field(
        description="Paths to intermediate processing files",
        default_factory=list
    )
    metadata: Dict[str, Any] = Field(
        description="Additional processing metadata",
        default_factory=dict
    )


class ProcessingPipeline(BaseModel):
    """Image processing pipeline configuration"""
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Pipeline name")
    description: str = Field(description="Pipeline description")
    steps: List[ProcessingAction] = Field(description="Ordered processing steps")
    target_conditions: List[DocumentCondition] = Field(
        description="Conditions this pipeline addresses"
    )
    expected_improvement: float = Field(
        description="Expected quality improvement (0-1)",
        ge=0, le=1
    )
    processing_time_estimate: float = Field(
        description="Estimated processing time in seconds"
    )
    prerequisites: List[DocumentCondition] = Field(
        description="Required conditions for this pipeline",
        default_factory=list
    )
    fallback_pipeline: Optional[str] = Field(
        default=None,
        description="Fallback pipeline ID if this one fails"
    )


class DocumentPreprocessingResult(BaseModel):
    """Complete document preprocessing result"""
    document_id: str = Field(description="Document identifier")
    original_path: str = Field(description="Original document path")
    processed_path: str = Field(description="Final processed document path")
    assessment: DocumentConditionAssessment = Field(
        description="Initial condition assessment"
    )
    processing_results: List[ImageProcessingResult] = Field(
        description="Results of each processing step"
    )
    final_quality_score: float = Field(
        description="Final quality score (0-1)",
        ge=0, le=1
    )
    total_processing_time: float = Field(
        description="Total processing time in seconds"
    )
    success: bool = Field(description="Overall preprocessing success")
    created_at: datetime = Field(
        description="Creation timestamp",
        default_factory=datetime.now
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata",
        default_factory=dict
    )