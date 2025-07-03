import os
import base64
import logging
from typing import Optional, List, Dict, Any
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from ..pydantic_models.document_condition_models import (
    DocumentConditionAssessment, 
    ConditionDetection,
    DocumentCondition,
    ProcessingAction,
    SeverityLevel
)

logger = logging.getLogger(__name__)


class DocumentConditionAnalyzer(Runnable):
    """Enhanced document condition analyzer with improved orientation detection"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            max_tokens=4096,
            temperature=0.1
        )
        
        # Create structured output analyzer
        self.condition_analyzer = self.llm.with_structured_output(DocumentConditionAssessment)
    
    def invoke(self, input_data: dict, config=None) -> DocumentConditionAssessment:
        """
        LangChain Runnable invoke method
        
        Args:
            input_data: Dict with 'image_path' key
            config: Optional configuration
            
        Returns:
            DocumentConditionAssessment: Enhanced condition analysis result
        """
        image_path = input_data.get('image_path')
        if not image_path:
            raise ValueError("Input must contain 'image_path' key")
        
        return self.analyze_document_condition(image_path)
    
    def analyze_document_condition(self, image_path: str) -> DocumentConditionAssessment:
        """Enhanced analysis with proper validation order - REFACTORED"""
        
        logger.info(f"Analyzing document condition for: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with Image.open(image_path) as img:
            image_dimensions = {"width": img.width, "height": img.height}
            image_mode = img.mode
        
        base64_image = self._encode_image(image_path)
        prompt = self._create_enhanced_condition_analysis_prompt(image_dimensions, image_mode)
        
        try:
            result = self._analyze_condition_structured(base64_image, prompt)
            
            # STEP 1: Initial enhancement (builds first set of recommendations)
            result = self._validate_and_enhance_assessment(result)
            
            # STEP 2: Orientation validation (may rebuild recommendations if corrections made)
            validation_results = self._validate_orientation_detection(result)
            
            # Debug logging
            self._log_final_results(result, validation_results)
            
            return result
        
        except Exception as e:
            logger.error(f"Enhanced condition analysis failed: {str(e)}")
            return self._create_enhanced_fallback_assessment(image_path)
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_enhanced_condition_analysis_prompt(self, image_dimensions: dict, image_mode: str) -> str:
        """Create enhanced prompt with improved orientation detection guidance"""
    
        width, height = image_dimensions["width"], image_dimensions["height"]
    
        prompt = f"""
Analyze this scanned document image comprehensively for preprocessing requirements with ENHANCED ORIENTATION DETECTION.

Image specifications:
- Dimensions: {width} × {height} pixels
- Color mode: {image_mode}
- Analysis focus: Document condition assessment with precise orientation detection

CRITICAL ORIENTATION ANALYSIS (HIGHEST PRIORITY):

**FUNDAMENTAL PRINCIPLE: MAJORITY TEXT ORIENTATION DETERMINES DOCUMENT ORIENTATION**

**STEP 1: PRIMARY ORIENTATION ASSESSMENT - MANDATORY SEQUENCE**

1. **INVERTED Detection (180° upside down) - ANALYZE THIS FIRST AND FOREMOST:**

   **PRIMARY INDICATORS (Check these BEFORE considering rotation):**
   - **Text Direction Test**: Are the MAJORITY of text lines readable horizontally but upside down?
     * Look for words that appear backwards/inverted when reading left-to-right
     * Text should appear horizontal (not vertical) but upside down
     * Example: "HELLO" would appear as "OӜƎH" (mirrored and inverted)
   
   - **Document Structure Inversion**: 
     * Headers/letterheads at BOTTOM instead of TOP
     * Page numbers in wrong position (e.g., "1 egaP" at top)
     * Signatures/date lines at TOP instead of BOTTOM
     * Natural document flow is bottom-to-top instead of top-to-bottom
   
   - **Content Organization**:
     * Title/heading content appears at bottom of image
     * Footer information appears at top of image
     * Paragraphs read naturally when image is rotated 180°
   
   **CRITICAL: If MAJORITY of text is horizontal but upside down → INVERTED (not rotated)**

2. **90° ROTATION Detection - ONLY AFTER RULING OUT INVERSION:**

   **Key Distinction**: Text runs VERTICALLY (top-to-bottom or bottom-to-top)
   
   - **ROTATED_CLOCKWISE_90 (needs 90° CCW to fix):**
     * MAJORITY of text lines run vertically from top to bottom
     * Normal reading requires turning head 90° counter-clockwise
     * Headers would be on the LEFT side of image
   
   - **ROTATED_CLOCKWISE_270 (needs 90° CW to fix):**
     * MAJORITY of text lines run vertically from bottom to top  
     * Normal reading requires turning head 90° clockwise
     * Headers would be on the RIGHT side of image
   
   **CRITICAL: Only classify as rotated if MAJORITY of main content is vertical**

3. **SKEWED Detection - ONLY if document is properly oriented:**
   - Text lines are horizontal and right-side up but tilted at small angles (1-15°)
   - Content orientation is correct, just misaligned

**DECISION MATRIX - FOLLOW THIS EXACT LOGIC:**

```
IF majority_text_is_horizontal AND appears_upside_down:
    → INVERTED (even if some vertical elements exist)

ELIF majority_text_is_vertical AND main_content_sideways:
    IF text_flows_top_to_bottom:
        → ROTATED_CLOCKWISE_90
    ELIF text_flows_bottom_to_top:
        → ROTATED_CLOCKWISE_270

ELIF majority_text_is_horizontal AND right_side_up AND tilted:
    → SKEWED

ELSE:
    → No orientation issue
```

**DISAMBIGUATION RULES:**

1. **Mixed Orientation Elements**: 
   - Ignore minor vertical elements (page numbers, margins notes, stamps)
   - Focus on MAIN BODY TEXT and PRIMARY CONTENT
   - Decide based on what constitutes 70%+ of readable content

2. **Priority Order**:
   - INVERTED detection has HIGHEST priority
   - Only consider rotation if inversion is ruled out
   - Never assign both INVERTED and ROTATED to same document

3. **Evidence Weighting**:
   - Main body text orientation: 60% weight
   - Header/footer position: 25% weight  
   - Document structure flow: 15% weight

**ENHANCED VALIDATION QUESTIONS:**
1. "If I rotate this image 180°, does the majority of text become readable?"
2. "Are headers/titles currently at the bottom when they should be at top?"
3. "Is the main body text horizontal but upside down, or actually vertical?"
4. "What percentage of total text content is oriented each way?"

**STEP 2: COMPREHENSIVE CONDITION ANALYSIS**

[Rest of analysis for quality, artifacts, etc. - keep existing content]

**CONFIDENCE SCORING FOR ORIENTATION:**
- 0.95-1.0: Clear majority evidence with multiple confirming indicators
- 0.85-0.94: Strong evidence from main content orientation
- 0.70-0.84: Good evidence but some conflicting elements
- 0.50-0.69: Mixed evidence, decision based on majority rule
- Below 0.50: Insufficient evidence, mark as uncertain

**PROCESSING RECOMMENDATIONS:**
Priority order for orientation corrections:
1. FLIP_180 (for inverted documents)
2. ROTATE_90_CW or ROTATE_90_CCW (for rotated documents)  
3. DESKEW (for skewed documents)
4. [Other quality improvements...]

**CRITICAL VALIDATION CHECKLIST:**
- [ ] Did I check for inverted horizontal text BEFORE considering rotation?
- [ ] Am I basing decision on MAJORITY content, not minority elements?
- [ ] Did I consider document structure (headers/footers) placement?
- [ ] Is my confidence score justified by the evidence?
- [ ] Did I provide specific evidence for my orientation decision?

The response must conform exactly to the DocumentConditionAssessment schema.
Focus on MAJORITY text orientation and provide detailed evidence for orientation decisions.
"""
        return prompt
    
    def _analyze_condition_structured(self, base64_image: str, prompt: str) -> DocumentConditionAssessment:
        """Perform enhanced structured condition analysis using LLM"""
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        )
        
        try:
            result: DocumentConditionAssessment = self.condition_analyzer.invoke([message])
            result = self._validate_and_enhance_assessment(result)
            return result
            
        except Exception as e:
            logger.error(f"Structured condition analysis failed: {str(e)}")
            raise ValueError(f"Condition analysis error: {str(e)}")
    
    def _validate_orientation_detection(self, assessment: DocumentConditionAssessment) -> Dict[str, Any]:
        """Enhanced validation with clear rotation angle detection - REFACTORED"""
        
        validation_results = {
            "has_orientation_conflict": False,
            "conflicting_conditions": [],
            "recommendations": [],
            "corrected_orientation": None
        }
        
        rotation_types = {
            DocumentCondition.ROTATED_180,
            DocumentCondition.ROTATED_90_CW,
            DocumentCondition.ROTATED_90_CCW,
            DocumentCondition.SKEWED
        }
        
        all_issues = assessment.primary_issues + assessment.secondary_issues
        rotation_conditions = [issue for issue in all_issues if issue.condition in rotation_types]
        
        # Track if we made any changes that require rebuilding recommendations
        recommendations_need_rebuild = False
        
        # Check for common misclassification: 90° rotation when should be 180° rotation
        rotated_90_issue = None
        rotated_180_issue = None
        
        for issue in rotation_conditions:
            if issue.condition in {DocumentCondition.ROTATED_90_CW, DocumentCondition.ROTATED_90_CCW}:
                rotated_90_issue = issue
            elif issue.condition == DocumentCondition.ROTATED_180:
                rotated_180_issue = issue
        
        # Special handling for 90° vs 180° rotation confusion
        if rotated_90_issue and not rotated_180_issue:
            # Check evidence for 180° rotation indicators
            evidence_lower = rotated_90_issue.evidence.lower()
            upside_down_indicators = [
                "upside down", "bottom", "header", "letterhead", 
                "signature", "footer", "page number", "inverted"
            ]
            
            horizontal_text_indicators = [
                "horizontal", "left to right", "line", "paragraph", "text line"
            ]
            
            upside_down_score = sum(1 for indicator in upside_down_indicators if indicator in evidence_lower)
            horizontal_score = sum(1 for indicator in horizontal_text_indicators if indicator in evidence_lower)
            
            # If evidence suggests 180° rotation rather than 90° rotation, flag for correction
            if upside_down_score >= 2 or (upside_down_score >= 1 and horizontal_score >= 1):
                validation_results["has_orientation_conflict"] = True
                validation_results["conflicting_conditions"].append(
                    f"Detected as {rotated_90_issue.condition.value} but evidence suggests ROTATED_180: {rotated_90_issue.evidence[:100]}..."
                )
                
                # Create corrected 180° rotation condition
                corrected_180 = ConditionDetection(
                    condition=DocumentCondition.ROTATED_180,
                    severity=rotated_90_issue.severity,
                    confidence=min(0.9, rotated_90_issue.confidence + 0.1),
                    evidence=f"Corrected from {rotated_90_issue.condition.value}: {rotated_90_issue.evidence}",
                    orientation_indicators=["horizontal_text_upside_down", "document_structure_rotated_180"],
                    detection_reasoning="Auto-corrected: Evidence suggests horizontal text upside down rather than vertical text",
                    recommended_actions=[ProcessingAction.ROTATE_180]
                )
                
                # Replace the 90° rotation condition with 180° rotation
                assessment.primary_issues = [
                    issue if issue.condition not in {DocumentCondition.ROTATED_90_CW, DocumentCondition.ROTATED_90_CCW} 
                    else corrected_180 for issue in assessment.primary_issues
                ]
                assessment.secondary_issues = [
                    issue if issue.condition not in {DocumentCondition.ROTATED_90_CW, DocumentCondition.ROTATED_90_CCW} 
                    else corrected_180 for issue in assessment.secondary_issues
                ]
                
                validation_results["corrected_orientation"] = "ROTATED_180"
                validation_results["recommendations"].append(f"Auto-corrected {rotated_90_issue.condition.value} to ROTATED_180 based on evidence analysis")
                
                # Mark that we need to rebuild recommendations
                recommendations_need_rebuild = True
        
        # Handle multiple rotation conflicts
        if len(rotation_conditions) > 1:
            validation_results["has_orientation_conflict"] = True
            validation_results["conflicting_conditions"] = [
                f"{cond.condition.value} (conf: {cond.confidence:.2f})" 
                for cond in rotation_conditions
            ]
            
            # Priority: ROTATED_180 > ROTATED_90_CCW > ROTATED_90_CW > SKEWED
            priority_order = [
                DocumentCondition.ROTATED_180,
                DocumentCondition.ROTATED_90_CCW, 
                DocumentCondition.ROTATED_90_CW,
                DocumentCondition.SKEWED
            ]
            
            # Find highest priority condition
            selected_condition = None
            for condition_type in priority_order:
                for issue in rotation_conditions:
                    if issue.condition == condition_type:
                        selected_condition = issue
                        break
                if selected_condition:
                    break
            
            if not selected_condition:
                # Fallback to highest confidence
                selected_condition = max(rotation_conditions, key=lambda x: x.confidence)
            
            # Remove other rotation conditions
            assessment.primary_issues = [
                issue for issue in assessment.primary_issues 
                if issue.condition not in rotation_types or issue == selected_condition
            ]
            assessment.secondary_issues = [
                issue for issue in assessment.secondary_issues 
                if issue.condition not in rotation_types or issue == selected_condition
            ]
            
            validation_results["recommendations"].append(
                f"Selected {selected_condition.condition.value} based on priority order"
            )
            
            # Mark that we need to rebuild recommendations
            recommendations_need_rebuild = True
        
        # CRITICAL: Rebuild processing recommendations if we made any changes
        if recommendations_need_rebuild:
            assessment.processing_recommendations = self._build_processing_recommendations(assessment)
            logger.info("🔄 Rebuilt processing recommendations after validation corrections")
        
        return validation_results

    def _build_processing_recommendations(self, assessment: DocumentConditionAssessment) -> List[ProcessingAction]:
        """Build ordered processing recommendations from detected issues"""
        
        # Collect all recommended actions from current issues
        all_actions = set()
        for issue in assessment.primary_issues + assessment.secondary_issues:
            all_actions.update(issue.recommended_actions)
        
        # Apply priority ordering
        action_priority = {
            ProcessingAction.ROTATE_180: 1,        # Highest priority - fix upside down
            ProcessingAction.ROTATE_90_CW: 2,      # Fix 90° rotations
            ProcessingAction.ROTATE_90_CCW: 2,     # Fix 90° rotations  
            ProcessingAction.DESKEW: 3,            # Fix small angle skew
            ProcessingAction.STRAIGHTEN_PERSPECTIVE: 4,
            ProcessingAction.CROP_MARGINS: 5,
            ProcessingAction.REMOVE_SHADOWS: 6,
            ProcessingAction.ENHANCE_CONTRAST: 7,
            ProcessingAction.GAMMA_CORRECTION: 8,
            ProcessingAction.HISTOGRAM_EQUALIZATION: 9,
            ProcessingAction.DENOISE: 10,
            ProcessingAction.SHARPEN: 11,
            ProcessingAction.ADAPTIVE_THRESHOLD: 12,
            ProcessingAction.BINARIZE: 13,
        }
        
        sorted_actions = sorted(list(all_actions), key=lambda x: action_priority.get(x, 99))
        return sorted_actions
    
    def _validate_and_enhance_assessment(self, assessment: DocumentConditionAssessment) -> DocumentConditionAssessment:
        """Validate and enhance the assessment result - REFACTORED"""
        
        # Build processing recommendations using shared method
        assessment.processing_recommendations = self._build_processing_recommendations(assessment)
        
        # Calculate other metrics
        if not hasattr(assessment, 'estimated_success_rate') or assessment.estimated_success_rate == 0:
            assessment.estimated_success_rate = self._calculate_enhanced_success_rate(assessment)
        
        assessment.processing_priority = self._calculate_processing_priority(assessment)
        
        return assessment
    
    def _calculate_enhanced_success_rate(self, assessment: DocumentConditionAssessment) -> float:
        """Calculate enhanced success rate based on detected conditions"""
    
        severity_impact = {
            SeverityLevel.NONE: 0.0,
            SeverityLevel.MILD: 0.03,
            SeverityLevel.MODERATE: 0.12,
            SeverityLevel.SEVERE: 0.25,
            SeverityLevel.CRITICAL: 0.45
        }
        
        condition_modifiers = {
            # Rotation issues - generally easier to fix
            DocumentCondition.ROTATED_180: 0.8,
            DocumentCondition.ROTATED_90_CW: 0.8,
            DocumentCondition.ROTATED_90_CCW: 0.8,
            DocumentCondition.SKEWED: 0.9,
            
            # Quality issues - varying difficulty
            DocumentCondition.HANDWRITTEN: 1.2,
            DocumentCondition.POOR_QUALITY: 1.1,
            DocumentCondition.BLURRY: 1.3,
            DocumentCondition.LOW_CONTRAST: 1.0,
            DocumentCondition.SHADOW_DISTORTION: 1.1,
        }
        
        total_impact = 0.0
        
        for issue in assessment.primary_issues:
            base_impact = severity_impact.get(issue.severity, 0.2)
            modifier = condition_modifiers.get(issue.condition, 1.0)
            total_impact += base_impact * modifier * issue.confidence
        
        for issue in assessment.secondary_issues:
            base_impact = severity_impact.get(issue.severity, 0.1) * 0.6
            modifier = condition_modifiers.get(issue.condition, 1.0)
            total_impact += base_impact * modifier * issue.confidence
        
        total_impact = min(total_impact, 0.7)
        success_rate = max(0.25, 1.0 - total_impact)
        
        return success_rate
    
    def _calculate_processing_priority(self, assessment: DocumentConditionAssessment) -> int:
        """Calculate processing priority (1=highest, 5=lowest)"""
        
        has_critical = any(issue.severity == SeverityLevel.CRITICAL for issue in assessment.primary_issues)
        has_severe = any(issue.severity == SeverityLevel.SEVERE for issue in assessment.primary_issues)
        
        rotation_types = {
            DocumentCondition.ROTATED_180,
            DocumentCondition.ROTATED_90_CW,
            DocumentCondition.ROTATED_90_CCW
        }
        has_rotation_issue = any(
            issue.condition in rotation_types for issue in assessment.primary_issues
        )
        
        if has_critical: return 1
        if has_severe or has_rotation_issue: return 2
        if len(assessment.primary_issues) > 2: return 3
        if len(assessment.primary_issues) > 0: return 4
        return 5
    
    def _create_enhanced_fallback_assessment(self, image_path: str) -> DocumentConditionAssessment:
        """Create enhanced fallback assessment when structured analysis fails"""
        
        logger.warning("Creating enhanced fallback condition assessment")
        
        fallback_condition = ConditionDetection(
            condition=DocumentCondition.POOR_QUALITY,
            severity=SeverityLevel.MODERATE,
            confidence=0.3,
            evidence="Automated analysis failed, manual inspection recommended",
            orientation_indicators=[],
            alternative_orientations=[],
            detection_reasoning="Structured analysis failed, using conservative fallback",
            recommended_actions=[
                ProcessingAction.ENHANCE_CONTRAST,
                ProcessingAction.DENOISE,
                ProcessingAction.SHARPEN
            ]
        )
        
        return DocumentConditionAssessment(
            overall_quality="Unable to assess - analysis failed",
            primary_issues=[fallback_condition],
            secondary_issues=[],
            processing_priority=3,
            estimated_success_rate=0.5,
            processing_recommendations=[
                ProcessingAction.ENHANCE_CONTRAST,
                ProcessingAction.DENOISE,
                ProcessingAction.SHARPEN
            ],
            special_handling_notes="Enhanced structured analysis failed, using conservative preprocessing approach",
            metadata={
                "fallback_used": True,
                "original_image_path": image_path,
                "analysis_failure": True,
                "enhanced_analyzer_version": "2.3"
            }
        )

    def batch_analyze(self, image_paths: List[str]) -> List[DocumentConditionAssessment]:
        """Analyze multiple document images with enhanced orientation detection"""
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Analyzing document {i+1}/{len(image_paths)}: {image_path}")
                assessment = self.analyze_document_condition(image_path)
                results.append(assessment)
            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {str(e)}")
                fallback = self._create_enhanced_fallback_assessment(image_path)
                results.append(fallback)
        return results
    
    def get_enhanced_analysis_summary(self, assessment: DocumentConditionAssessment) -> Dict[str, Any]:
        """Get enhanced human-readable summary of the condition analysis"""
        
        summary = {
            "overall_quality": assessment.overall_quality,
            "priority": assessment.processing_priority,
            "success_rate": f"{assessment.estimated_success_rate:.1%}",
            "primary_issues": [],
            "orientation_detected": None,
            "recommended_actions": [action.value for action in assessment.processing_recommendations],
            "processing_complexity": "low",
        }
        
        orientation_types = {
            DocumentCondition.INVERTED,
            DocumentCondition.ROTATED_CLOCKWISE_90,
            DocumentCondition.ROTATED_CLOCKWISE_270,
            DocumentCondition.SKEWED
        }
        
        for issue in assessment.primary_issues:
            issue_summary = {
                "condition": issue.condition.value,
                "severity": issue.severity.value,
                "confidence": f"{issue.confidence:.2f}",
                "evidence": issue.evidence
            }
            summary["primary_issues"].append(issue_summary)
            
            if issue.condition in orientation_types:
                summary["orientation_detected"] = issue.condition.value
        
        action_count = len(assessment.processing_recommendations)
        if action_count > 6:
            summary["processing_complexity"] = "high"
        elif action_count > 3:
            summary["processing_complexity"] = "medium"
        
        return summary
    
    def _log_final_results(self, result: DocumentConditionAssessment, validation_results: Dict[str, Any]):
        """Centralized logging for final results"""
        
        # Log detected orientations
        orientation_types = {
            DocumentCondition.ROTATED_180,
            DocumentCondition.ROTATED_90_CW,
            DocumentCondition.ROTATED_90_CCW,
            DocumentCondition.SKEWED
        }
        
        detected_orientations = []
        for issue in result.primary_issues + result.secondary_issues:
            if issue.condition in orientation_types:
                detected_orientations.append({
                    'condition': issue.condition.value,
                    'confidence': issue.confidence,
                    'evidence': issue.evidence[:200],
                    'reasoning': getattr(issue, 'detection_reasoning', 'N/A')
                })
        
        if detected_orientations:
            logger.info("=== FINAL ORIENTATION DETECTION ===")
            for orient in detected_orientations:
                logger.info(f"Final: {orient['condition']} (conf: {orient['confidence']:.2f})")
                logger.info(f"Evidence: {orient['evidence']}")
        
        # Log final processing recommendations
        logger.info(f"=== FINAL PROCESSING RECOMMENDATIONS ===")
        for i, action in enumerate(result.processing_recommendations, 1):
            logger.info(f"{i}. {action.value}")
        
        # Log validation results
        if validation_results["has_orientation_conflict"]:
            logger.warning("=== ORIENTATION CONFLICTS DETECTED ===")
            for conflict in validation_results["conflicting_conditions"]:
                logger.warning(f"Conflict: {conflict}")
            for rec in validation_results["recommendations"]:
                logger.info(f"Correction: {rec}")
        
        if validation_results.get("corrected_orientation"):
            logger.info(f"Auto-corrected orientation to: {validation_results['corrected_orientation']}")
        
        logger.info(f"Final assessment: {len(result.primary_issues)} primary issues, {len(result.secondary_issues)} secondary issues")
