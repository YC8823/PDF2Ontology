from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime

class ProcessingStage(str, Enum):
    """Processing stage enumeration"""
    VISUAL_ANALYSIS = "visual_analysis"
    TEXT_EXTRACTION = "text_extraction"
    TABLE_PROCESSING = "table_processing"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    GRAPH_CONSTRUCTION = "graph_construction"
    VALIDATION = "validation"
    EXPORT = "export"

class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StageResult(BaseModel):
    """Processing stage result model"""
    stage: ProcessingStage = Field(description="Processing stage")
    status: ProcessingStatus = Field(description="Stage status")
    start_time: datetime = Field(description="Stage start time")
    end_time: Optional[datetime] = Field(description="Stage end time", default=None)
    duration: Optional[float] = Field(
        description="Duration in seconds", 
        default=None
    )
    success: bool = Field(description="Whether stage completed successfully")
    error_message: Optional[str] = Field(
        description="Error message if failed", 
        default=None
    )
    warnings: List[str] = Field(description="Warning messages", default_factory=list)
    metrics: Dict[str, Any] = Field(
        description="Stage-specific metrics", 
        default_factory=dict
    )
    output_data: Optional[Dict[str, Any]] = Field(
        description="Stage output data", 
        default=None
    )

class PipelineResult(BaseModel):
    """Pipeline execution result model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_name: str = Field(description="Pipeline name")
    input_documents: List[str] = Field(description="Input document paths")
    start_time: datetime = Field(description="Pipeline start time")
    end_time: Optional[datetime] = Field(description="Pipeline end time", default=None)
    total_duration: Optional[float] = Field(
        description="Total duration in seconds", 
        default=None
    )
    status: ProcessingStatus = Field(description="Overall pipeline status")
    stage_results: List[StageResult] = Field(
        description="Results for each stage"
    )
    final_output: Optional[Dict[str, Any]] = Field(
        description="Final pipeline output", 
        default=None
    )
    error_summary: Optional[str] = Field(
        description="Error summary if failed", 
        default=None
    )
    performance_metrics: Dict[str, float] = Field(
        description="Performance metrics", 
        default_factory=dict
    )
    resource_usage: Dict[str, Any] = Field(
        description="Resource usage statistics", 
        default_factory=dict
    )
    configuration: Dict[str, Any] = Field(
        description="Pipeline configuration used", 
        default_factory=dict
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata", 
        default_factory=dict
    )
