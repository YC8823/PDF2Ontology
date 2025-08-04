"""
Refactored Document Condition Analyzer
File: src/analyzers/document_conditioin_cv_analyzer.py

Streamlined document condition analyzer using modular detection utilities
"""

import cv2
import numpy as np
import os
import logging
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

# Import existing models and utilities
from ..pydantic_models.document_condition_models import (
    DocumentCondition, ProcessingAction, SeverityLevel, 
    ConditionDetection, DocumentConditionAssessment,
    ImageProcessingResult, DocumentPreprocessingResult
)
from ..utils.image_utils import ImageProcessor, convert_pdf_to_images

# Import the new modular analysis utilities
from ..utils.document_analysis_utils import (
    RotationDetector, SkewDetector, QualityAssessor, ContentAnalyzer,
    convert_rotation_result_to_condition, convert_skew_result_to_condition,
    convert_quality_result_to_conditions, convert_content_result_to_conditions
)

logger = logging.getLogger(__name__)


# =============================================================================
# REFACTORED DOCUMENT CONDITION ANALYZER
# =============================================================================

class DocumentConditionAnalyzer:
    """
    Refactored document condition analyzer using modular detection utilities
    Now serves as a coordinator for specialized analyzers
    """
    
    def __init__(self, confidence_threshold: float = 0.7, enable_debug: bool = False):
        """
        Initialize the document condition analyzer
        
        Args:
            confidence_threshold: Minimum confidence threshold for detections
            enable_debug: Enable debug mode for detailed logging
        """
        self.confidence_threshold = confidence_threshold
        self.enable_debug = enable_debug
        
        # Initialize specialized analyzers
        self.rotation_detector = RotationDetector(confidence_threshold)
        self.skew_detector = SkewDetector(confidence_threshold)
        self.quality_assessor = QualityAssessor(confidence_threshold)
        self.content_analyzer = ContentAnalyzer(confidence_threshold)
        
        # Set debug mode for all analyzers
        if enable_debug:
            self.rotation_detector.set_debug_mode(True)
            self.skew_detector.set_debug_mode(True)
            self.quality_assessor.set_debug_mode(True)
            self.content_analyzer.set_debug_mode(True)
        
        # Store analysis results for debugging
        self.analysis_results = {}
    
    def analyze_document_condition(self, image_path: str) -> DocumentConditionAssessment:
        """
        Comprehensive document condition analysis using modular CV methods
        
        Args:
            image_path: Path to document image
            
        Returns:
            DocumentConditionAssessment: Complete analysis result
        """
        logger.info(f"Starting document condition analysis for: {image_path}")
        
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Store original image info for debugging
        self.original_image = image.copy()
        self.image_height, self.image_width = image.shape[:2]
        
        # Run all modular analyses
        detected_conditions = []
        
        try:
            # 1. Rotation Detection (Primary Focus)
            rotation_result = self.rotation_detector.analyze(image)
            self.analysis_results['rotation'] = rotation_result
            
            rotation_condition = convert_rotation_result_to_condition(rotation_result)
            if rotation_condition:
                detected_conditions.append(rotation_condition)
                logger.info(f"Rotation detected: {rotation_result.rotation_angle}° (confidence: {rotation_result.confidence:.2f})")
            
            # 2. Skew Detection
            skew_result = self.skew_detector.analyze(image)
            self.analysis_results['skew'] = skew_result
            
            skew_condition = convert_skew_result_to_condition(skew_result)
            if skew_condition:
                detected_conditions.append(skew_condition)
                logger.info(f"Skew detected: {skew_result.skew_angle:.1f}° (confidence: {skew_result.confidence:.2f})")
            
            # 3. Quality Assessment
            quality_result = self.quality_assessor.analyze(image)
            self.analysis_results['quality'] = quality_result
            
            quality_conditions = convert_quality_result_to_conditions(quality_result)
            detected_conditions.extend(quality_conditions)
            if quality_conditions:
                logger.info(f"Quality issues detected: {[c.condition.value for c in quality_conditions]}")
            
            # 4. Content Analysis
            content_result = self.content_analyzer.analyze(image)
            self.analysis_results['content'] = content_result
            
            content_conditions = convert_content_result_to_conditions(content_result)
            detected_conditions.extend(content_conditions)
            if content_conditions:
                logger.info(f"Content features detected: {[c.condition.value for c in content_conditions]}")
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            # Continue with whatever conditions were detected
        
        # Generate comprehensive assessment
        assessment = self._generate_assessment(detected_conditions)
        
        logger.info(f"Analysis complete. Found {len(detected_conditions)} conditions. Overall quality: {assessment.overall_quality}")
        return assessment
    
    def _generate_assessment(self, conditions: List[ConditionDetection]) -> DocumentConditionAssessment:
        """
        Generate comprehensive assessment from detected conditions
        Simplified version that focuses on the core logic
        """
        if not conditions:
            return DocumentConditionAssessment(
                overall_quality="Good",
                primary_issues=[],
                secondary_issues=[],
                processing_priority=5,
                estimated_success_rate=0.9,
                processing_recommendations=[ProcessingAction.NO_ACTION],
                special_handling_notes="Document appears to be in good condition"
            )
        
        # Categorize conditions by severity
        critical_conditions = [c for c in conditions if c.severity == SeverityLevel.SEVERE]
        moderate_conditions = [c for c in conditions if c.severity == SeverityLevel.MODERATE]
        mild_conditions = [c for c in conditions if c.severity == SeverityLevel.MILD]
        
        # Determine overall quality and processing priority
        if critical_conditions:
            overall_quality = "Poor"
            priority = 1
            base_success_rate = 0.4
        elif moderate_conditions:
            overall_quality = "Fair"
            priority = 2 if len(moderate_conditions) > 2 else 3
            base_success_rate = 0.7
        else:
            overall_quality = "Good"
            priority = 4
            base_success_rate = 0.9
        
        # Adjust success rate based on specific conditions
        success_rate = base_success_rate
        rotation_issues = [c for c in conditions if c.condition in [
            DocumentCondition.ROTATED_CLOCKWISE_90, DocumentCondition.ROTATED_CLOCKWISE_270, DocumentCondition.ROTATED_180
        ]]
        
        if rotation_issues:
            # Rotation issues are fixable, so don't penalize success rate too much
            success_rate = max(success_rate, 0.6)
        
        # Generate processing recommendations with optimal ordering
        processing_recommendations = self._generate_processing_recommendations(conditions)
        
        # Generate special handling notes
        special_notes = self._generate_special_notes(conditions)
        
        # Get quality_result object for metadata
        quality_result = self.analysis_results.get('quality')
        
        return DocumentConditionAssessment(
            overall_quality=overall_quality,
            primary_issues=critical_conditions + moderate_conditions,
            secondary_issues=mild_conditions,
            processing_priority=priority,
            estimated_success_rate=success_rate,
            processing_recommendations=processing_recommendations,
            special_handling_notes=special_notes,
            metadata={
                "analysis_results": self._serialize_analysis_results(),
                "total_conditions": len(conditions),
                "rotation_detected": any(c.condition in [
                    DocumentCondition.ROTATED_CLOCKWISE_90, DocumentCondition.ROTATED_CLOCKWISE_270, 
                    DocumentCondition.ROTATED_180
                ] for c in conditions),
                # Use dot notation to access the object's attribute and provide a default
                "quality_score": quality_result.overall_score if quality_result else 0.5
            }
        )
    
    def _generate_processing_recommendations(self, conditions: List[ConditionDetection]) -> List[ProcessingAction]:
        """
        Generate optimized processing recommendations
        """
        all_actions = []
        for condition in conditions:
            all_actions.extend(condition.recommended_actions)
        
        if not all_actions:
            return [ProcessingAction.NO_ACTION]
        
        # Remove duplicates while preserving order
        unique_actions = []
        for action in all_actions:
            if action not in unique_actions:
                unique_actions.append(action)
        
        # Apply priority ordering (same as ProcessingStrategyGenerator)
        priority_groups = {
            1: [ProcessingAction.ROTATE_180, ProcessingAction.ROTATE_90_CW, ProcessingAction.ROTATE_90_CCW],
            2: [ProcessingAction.DESKEW],
            3: [ProcessingAction.CROP_MARGINS],
            4: [ProcessingAction.REMOVE_SHADOWS],
            5: [ProcessingAction.ENHANCE_CONTRAST, ProcessingAction.GAMMA_CORRECTION],
            6: [ProcessingAction.DENOISE],
            7: [ProcessingAction.SHARPEN],
            8: [ProcessingAction.BINARIZE, ProcessingAction.ADAPTIVE_THRESHOLD]
        }
        
        # Order actions by priority
        ordered_actions = []
        for priority in sorted(priority_groups.keys()):
            for action in unique_actions:
                if action in priority_groups[priority] and action not in ordered_actions:
                    ordered_actions.append(action)
        
        # Add any remaining actions
        for action in unique_actions:
            if action not in ordered_actions:
                ordered_actions.append(action)
        
        return ordered_actions
    
    def _generate_special_notes(self, conditions: List[ConditionDetection]) -> Optional[str]:
        """
        Generate special handling notes based on conditions
        """
        notes = []
        
        # Check for specific condition types
        condition_types = [c.condition for c in conditions]
        
        if DocumentCondition.HANDWRITTEN in condition_types:
            notes.append("Contains handwritten content - may require specialized OCR settings")
        
        if DocumentCondition.MULTI_COLUMN in condition_types:
            notes.append("Multi-column layout detected - consider column-aware processing")
        
        if DocumentCondition.COMPLEX_LAYOUT in condition_types:
            notes.append("Complex layout detected - may require manual verification")
        
        # Check for rotation issues
        rotation_conditions = [c for c in conditions if c.condition in [
            DocumentCondition.ROTATED_CLOCKWISE_90, DocumentCondition.ROTATED_CLOCKWISE_270, 
            DocumentCondition.ROTATED_180
        ]]
        
        if rotation_conditions:
            rotation_angles = []
            for rc in rotation_conditions:
                if rc.condition == DocumentCondition.ROTATED_CLOCKWISE_90:
                    rotation_angles.append("90° CW")
                elif rc.condition == DocumentCondition.ROTATED_CLOCKWISE_270:
                    rotation_angles.append("90° CCW")
                elif rc.condition == DocumentCondition.ROTATED_180:
                    rotation_angles.append("180°")
            
            if rotation_angles:
                notes.append(f"Rotation correction needed: {', '.join(rotation_angles)}")
        
        # Check for quality issues
        quality_conditions = [c for c in conditions if c.condition in [
            DocumentCondition.POOR_QUALITY, DocumentCondition.BLURRY, 
            DocumentCondition.LOW_CONTRAST, DocumentCondition.DARK_BACKGROUND
        ]]
        
        if len(quality_conditions) > 2:
            notes.append("Multiple quality issues detected - may require careful preprocessing")
        
        return "; ".join(notes) if notes else None
    
    def _serialize_analysis_results(self) -> Dict[str, Any]:
        """
        Serialize analysis results for metadata storage
        """
        serialized = {}
        
        for key, result in self.analysis_results.items():
            if hasattr(result, '__dict__'):
                serialized[key] = result.__dict__
            else:
                serialized[key] = str(result)
        
        return serialized
    
    def get_detailed_analysis_report(self) -> Dict[str, Any]:
        """
        Get detailed analysis report for debugging and monitoring
        """
        report = {
            "image_info": {
                "width": self.image_width,
                "height": self.image_height,
                "aspect_ratio": self.image_width / self.image_height
            },
            "analysis_results": self.analysis_results,
            "timestamp": datetime.now().isoformat()
        }
        
        return report


# =============================================================================
# PROCESSING STRATEGY GENERATOR (Simplified)
# =============================================================================

class ProcessingStrategyGenerator:
    """
    Simplified processing strategy generator
    """
    
    @staticmethod
    def generate_processing_strategy(assessment: DocumentConditionAssessment) -> List[ProcessingAction]:
        """
        Generate processing strategy - now simply returns the assessment recommendations
        which are already optimally ordered
        """
        return assessment.processing_recommendations


# =============================================================================
# MAIN DOCUMENT PREPROCESSOR (Updated)
# =============================================================================

class DocumentPreprocessor:
    """
    Main document preprocessing workflow orchestrator
    Updated to use the refactored analyzer
    """
    
    def __init__(self, output_dir: str = "preprocessed_documents", 
                 temp_dir: Optional[str] = None,
                 enable_debug: bool = False):
        """
        Initialize the document preprocessor
        
        Args:
            output_dir: Output directory for processed files
            temp_dir: Temporary directory for processing
            enable_debug: Enable debug mode for detailed analysis
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.enable_debug = enable_debug
        
        # Initialize components with debug mode
        self.condition_analyzer = DocumentConditionAnalyzer(
            confidence_threshold=0.7, 
            enable_debug=enable_debug
        )
        self.image_processor = ImageProcessor(temp_dir=self.temp_dir)
        self.strategy_generator = ProcessingStrategyGenerator()
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.enable_debug else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'preprocessing.log'),
                logging.StreamHandler()
            ]
        )
    
    def process_pdf(self, pdf_path: str, 
                   document_id: Optional[str] = None) -> DocumentPreprocessingResult:
        """
        Complete PDF preprocessing workflow
        
        Args:
            pdf_path: Path to input PDF file
            document_id: Optional document identifier
            
        Returns:
            DocumentPreprocessingResult: Complete preprocessing result
        """
        start_time = datetime.now()
        
        if document_id is None:
            document_id = f"doc_{int(start_time.timestamp())}"
        
        logger.info(f"Starting PDF preprocessing for document: {document_id}")
        
        try:
            # Step 1: Convert PDF to images
            logger.info("Converting PDF to images...")
            images = convert_pdf_to_images(pdf_path, dpi=300)
            
            if not images:
                raise ValueError("Failed to convert PDF to images")
            
            logger.info(f"Successfully converted PDF to {len(images)} images")
            
            # Step 2: Process each page
            all_processing_results = []
            processed_image_paths = []
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"Processing page {page_num}/{len(images)}")
                
                # Save image temporarily
                temp_image_path = os.path.join(self.temp_dir, f"page_{page_num}.png")
                image.save(temp_image_path)
                
                # Process single page
                page_result = self._process_single_page(
                    temp_image_path, page_num, document_id
                )
                
                all_processing_results.append(page_result)
                processed_image_paths.append(page_result.output_path)
            
            # Step 3: Combine results
            final_result = self._combine_results(
                document_id, pdf_path, all_processing_results, start_time
            )
            
            # Step 4: Save results
            self._save_results(final_result)
            
            logger.info(f"PDF preprocessing completed for document: {document_id}")
            return final_result
            
        except Exception as e:
            logger.error(f"PDF preprocessing failed: {str(e)}")
            raise
    
    def _process_single_page(self, image_path: str, page_num: int, 
                           document_id: str) -> ImageProcessingResult:
        """
        Process a single page through the complete workflow
        """
        logger.info(f"  Analyzing conditions for page {page_num}")
        
        # Step 1: Analyze document conditions using refactored analyzer
        assessment = self.condition_analyzer.analyze_document_condition(image_path)
        
        # Step 2: Generate processing strategy
        processing_actions = self.strategy_generator.generate_processing_strategy(assessment)
        
        # --- START OF FIX ---
        # Log planned actions safely, handling both Enum and str types
        action_values = [a.value if isinstance(a, ProcessingAction) else a for a in processing_actions]
        logger.info(f"  Planned actions for page {page_num}: {action_values}")
        # --- END OF FIX ---
        
        # Step 3: Apply processing
        if processing_actions and processing_actions[0] != ProcessingAction.NO_ACTION:
            result = self.image_processor.preprocess_image(
                image_path=image_path,
                processing_level="custom",
                custom_actions=processing_actions
            )
        else:
            # No processing needed, just copy the file
            output_path = self._copy_image_to_output(image_path, page_num, document_id)
            result = ImageProcessingResult(
                input_path=image_path,
                output_path=output_path,
                actions_applied=[ProcessingAction.NO_ACTION],
                success=True,
                processing_time=0.0,
                before_conditions=[c.condition for c in assessment.primary_issues],
                after_conditions=[]
            )
        
        # Add assessment info to result metadata
        result.metadata.update({
            "page_number": page_num,
            "document_id": document_id,
            "condition_assessment": assessment.dict(),
            "overall_quality": assessment.overall_quality,
            "processing_priority": assessment.processing_priority,
            "detailed_analysis": self.condition_analyzer.get_detailed_analysis_report() if self.enable_debug else None
        })
        
        return result
    
    def _copy_image_to_output(self, image_path: str, page_num: int, 
                             document_id: str) -> str:
        """Copy image to output directory without processing"""
        output_filename = f"{document_id}_page_{page_num}_processed.png"
        output_path = self.output_dir / output_filename
        
        import shutil
        shutil.copy2(image_path, output_path)
        
        return str(output_path)
    
    def _combine_results(self, document_id: str, original_path: str,
                        page_results: List[ImageProcessingResult],
                        start_time: datetime) -> DocumentPreprocessingResult:
        """
        Combine all page processing results into final document result
        """
        # Calculate overall statistics
        total_processing_time = sum(r.processing_time for r in page_results)
        successful_pages = sum(1 for r in page_results if r.success)
        
        # Calculate quality score
        quality_improvements = [r.quality_improvement for r in page_results 
                              if r.quality_improvement is not None]
        final_quality_score = np.mean(quality_improvements) if quality_improvements else 0.7
        
        # Get first page's assessment for document-level info
        first_page_assessment_dict = page_results[0].metadata.get("condition_assessment")
        
        # Re-create the Pydantic model from the dictionary to ensure correct types
        if first_page_assessment_dict:
            first_page_assessment = DocumentConditionAssessment.model_validate(first_page_assessment_dict)
        else:
            first_page_assessment = None

        return DocumentPreprocessingResult(
            document_id=document_id,
            original_path=original_path,
            processed_path=str(self.output_dir / f"{document_id}_processed"),
            assessment=first_page_assessment,
            processing_results=page_results,
            final_quality_score=final_quality_score,
            total_processing_time=total_processing_time,
            success=successful_pages == len(page_results),
            created_at=start_time,
            metadata={
                "total_pages": len(page_results),
                "successful_pages": successful_pages,
                "failed_pages": len(page_results) - successful_pages,
                "output_directory": str(self.output_dir),
                "debug_mode": self.enable_debug
            }
        )
    
    def _save_results(self, result: DocumentPreprocessingResult):
        """Save processing results to JSON file"""
        output_file = self.output_dir / f"{result.document_id}_processing_report.json"
        
        # Convert to serializable format
        result_dict = result.dict()
        result_dict['created_at'] = result.created_at.isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Processing report saved to: {output_file}")
    
    def cleanup(self):
        """Clean up temporary files"""
        self.image_processor.cleanup_intermediate_files()
        
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# =============================================================================
# MAIN WORKFLOW FUNCTION 
# =============================================================================

def main_preprocessing_workflow(pdf_path: str, 
                               output_dir: str = "preprocessed_documents",
                               enable_debug: bool = False):
    """
    Main preprocessing workflow function
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Output directory for processed files
        enable_debug: Enable debug mode for detailed analysis
        
    Returns:
        DocumentPreprocessingResult: Complete processing result
    """
    
    # Initialize preprocessor with debug mode
    preprocessor = DocumentPreprocessor(
        output_dir=output_dir,
        enable_debug=enable_debug
    )
    
    try:
        # Process PDF
        result = preprocessor.process_pdf(pdf_path)
        
        # Print summary
        print(f"\n=== Processing Summary ===")
        print(f"Document ID: {result.document_id}")
        print(f"Total Pages: {result.metadata['total_pages']}")
        print(f"Successful Pages: {result.metadata['successful_pages']}")
        print(f"Final Quality Score: {result.final_quality_score:.2f}")
        print(f"Total Processing Time: {result.total_processing_time:.2f}s")
        print(f"Overall Success: {result.success}")
        
        # Show per-page results
        print(f"\n=== Per-Page Results ===")
        for i, page_result in enumerate(result.processing_results, 1):
            actions = [a.value if isinstance(a, ProcessingAction) else a for a in page_result.actions_applied]
            quality = page_result.quality_improvement
            overall_quality = page_result.metadata.get('overall_quality', 'N/A')
            
            print(f"Page {i}: {actions} - Quality: {overall_quality}")
            if quality is not None:
                print(f"  Quality improvement: {quality:.2f}")
        
        return result
        
    finally:
        # Clean up
        preprocessor.cleanup()


# Example usage with debug mode
if __name__ == "__main__":
    # Example usage with debug enabled
    pdf_path = "data/inputs/var3.pdf"
    
    if os.path.exists(pdf_path):
        try:
            # Enable debug mode to see detailed analysis
            result = main_preprocessing_workflow(
                pdf_path, 
                enable_debug=True  # Enable detailed logging
            )
            print(f"\nProcessing completed successfully!")
            print(f"Processed files saved to: {result.processed_path}")
            
        except Exception as e:
            print(f"Processing failed: {str(e)}")
    else:
        print(f"Sample PDF not found: {pdf_path}")
        print("Please provide a valid PDF file path.")
