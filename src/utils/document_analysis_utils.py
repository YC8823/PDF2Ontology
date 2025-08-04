"""
Document Analysis Utilities - Modular Detection Functions
File: src/utils/document_analysis_utils.py

Modular computer vision functions for document condition analysis
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..pydantic_models.document_condition_models import (
    DocumentCondition, ProcessingAction, SeverityLevel, ConditionDetection
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES FOR ANALYSIS RESULTS
# =============================================================================

@dataclass
class RotationAnalysisResult:
    """Result of rotation analysis"""
    rotation_angle: int  # 0, 90, 180, 270
    confidence: float
    evidence: str
    text_orientation_score: float
    edge_orientation_score: float
    aspect_ratio: float
    detection_method: str


@dataclass
class SkewAnalysisResult:
    """Result of skew analysis"""
    skew_angle: float
    confidence: float
    evidence: str
    line_count: int
    detection_method: str


@dataclass
class QualityAnalysisResult:
    """Result of image quality analysis"""
    overall_score: float
    contrast_score: float
    brightness_score: float
    sharpness_score: float
    noise_level: float
    detected_issues: List[str]
    recommendations: List[ProcessingAction]


@dataclass
class ContentAnalysisResult:
    """Result of content analysis"""
    has_handwriting: bool
    handwriting_confidence: float
    column_count: int
    layout_complexity: str
    detected_features: List[str]


# =============================================================================
# BASE ANALYZER CLASS
# =============================================================================

class BaseDocumentAnalyzer(ABC):
    """Base class for document analyzers"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.debug_mode = False
    
    def set_debug_mode(self, debug: bool):
        """Enable/disable debug mode"""
        self.debug_mode = debug
    
    @abstractmethod
    def analyze(self, image: np.ndarray) -> Any:
        """Analyze the image and return results"""
        pass
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """Validate input image"""
        if image is None:
            raise ValueError("Input image is None")
        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D or 3D array")
        return True


# =============================================================================
# ROTATION DETECTOR
# =============================================================================

class RotationDetector(BaseDocumentAnalyzer):
    """
    Specialized rotation detection using multiple CV techniques
    Primary focus on 90°, 180° rotation detection
    """
    
    def analyze(self, image: np.ndarray) -> RotationAnalysisResult:
        """
        Detect document rotation using multiple CV techniques
        
        Args:
            image: Input image as numpy array
            
        Returns:
            RotationAnalysisResult: Comprehensive rotation analysis
        """
        self._validate_image(image)
        
        logger.debug("Starting rotation detection analysis...")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Get image dimensions
        height, width = gray.shape
        aspect_ratio = width / height
        
        # Method 1: Text line orientation analysis
        text_orientation = self._analyze_text_orientation(gray)
        
        # Method 2: Edge-based orientation analysis
        edge_orientation = self._analyze_edge_orientation(gray)
        
        # Method 3: Combined analysis with aspect ratio
        rotation_angle, confidence, evidence = self._determine_rotation_angle(
            text_orientation, edge_orientation, aspect_ratio
        )
        
        return RotationAnalysisResult(
            rotation_angle=rotation_angle,
            confidence=confidence,
            evidence=evidence,
            text_orientation_score=text_orientation,
            edge_orientation_score=edge_orientation,
            aspect_ratio=aspect_ratio,
            detection_method="combined_cv_analysis"
        )
    
    def _analyze_text_orientation(self, gray: np.ndarray) -> float:
        """
        Analyze text orientation using connected components and morphological analysis
        """
        # Apply adaptive thresholding to highlight text
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find connected components (potential text regions)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Filter contours by size and aspect ratio (likely text)
        text_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 20000:  # Reasonable text size range
                # Additional filtering by aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # Reasonable text aspect ratio
                    text_contours.append(contour)
        
        if not text_contours:
            return 0.0
        
        # Analyze orientation of text regions
        orientation_scores = []
        for contour in text_contours[:100]:  # Limit analysis for performance
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            
            # Normalize angle to meaningful range
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            orientation_scores.append(angle)
        
        if not orientation_scores:
            return 0.0
        
        # Calculate dominant orientation using weighted histogram
        weights = [cv2.contourArea(contour) for contour in text_contours[:100]]
        if weights:
            dominant_angle = np.average(orientation_scores, weights=weights)
        else:
            dominant_angle = np.median(orientation_scores)
        
        return float(dominant_angle)
    
    def _analyze_edge_orientation(self, gray: np.ndarray) -> float:
        """
        Analyze dominant edge orientation using Hough line transform
        """
        # Apply edge detection with optimized parameters
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply morphological operations to enhance line detection
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=max(50, min(gray.shape) // 4))
        
        if lines is None:
            return 0.0
        
        # Analyze line angles with weighting
        angle_votes = []
        for rho, theta in lines[:200, 0]:  # Limit for performance
            angle = np.degrees(theta)
            # Convert to -90 to +90 range
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            
            # Weight by line strength (rho indicates line prominence)
            weight = abs(rho) / max(gray.shape)
            angle_votes.extend([angle] * int(weight * 10))
        
        if not angle_votes:
            return 0.0
        
        # Find dominant angle using histogram analysis
        hist, bin_edges = np.histogram(angle_votes, bins=36, range=(-90, 90))
        dominant_bin = np.argmax(hist)
        dominant_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
        
        return float(dominant_angle)
    
    def _determine_rotation_angle(self, text_orientation: float, edge_orientation: float, 
                                 aspect_ratio: float) -> Tuple[int, float, str]:
        """
        Determine the required rotation angle and confidence
        
        Returns:
            Tuple[rotation_angle, confidence, evidence]
        """
        # Weight different analyses (text analysis more reliable for documents)
        combined_angle = (text_orientation * 0.7 + edge_orientation * 0.3)
        
        # Determine rotation needed
        rotation_needed = 0
        confidence = 0.0
        evidence = ""
        
        # Check for rotations with improved logic
        if -15 <= combined_angle <= 15:
            rotation_needed = 0
            confidence = 0.9
            evidence = "Document appears correctly oriented"
        elif 15 < combined_angle <= 75:
            rotation_needed = 270  # Rotate 270° (or -90°) to correct
            confidence = 0.85
            evidence = f"Document rotated 90° clockwise (combined angle: {combined_angle:.1f}°)"
        elif 75 < combined_angle <= 105:
            rotation_needed = 270
            confidence = 0.9
            evidence = f"Document clearly rotated 90° clockwise (combined angle: {combined_angle:.1f}°)"
        elif -75 <= combined_angle < -15:
            rotation_needed = 90  # Rotate 90° to correct
            confidence = 0.85
            evidence = f"Document rotated 90° counter-clockwise (combined angle: {combined_angle:.1f}°)"
        elif -105 <= combined_angle < -75:
            rotation_needed = 90
            confidence = 0.9
            evidence = f"Document clearly rotated 90° counter-clockwise (combined angle: {combined_angle:.1f}°)"
        else:
            rotation_needed = 180
            confidence = 0.8
            evidence = f"Document appears upside down (combined angle: {combined_angle:.1f}°)"
        
        # Adjust confidence based on aspect ratio consistency
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:  # Likely orientation mismatch
            if rotation_needed in [90, 270]:
                confidence += 0.05
        
        # Boost confidence for consistent measurements
        if abs(text_orientation - edge_orientation) < 15:
            confidence += 0.05
        
        # Penalize if measurements are too inconsistent
        if abs(text_orientation - edge_orientation) > 45:
            confidence -= 0.1
        
        confidence = max(0.0, min(confidence, 1.0))
        
        return rotation_needed, confidence, evidence


# =============================================================================
# SKEW DETECTOR
# =============================================================================

class SkewDetector(BaseDocumentAnalyzer):
    """
    Specialized skew detection using Hough line analysis
    """
    
    def analyze(self, image: np.ndarray) -> SkewAnalysisResult:
        """
        Detect document skew using Hough line analysis
        
        Args:
            image: Input image as numpy array
            
        Returns:
            SkewAnalysisResult: Skew detection results
        """
        self._validate_image(image)
        
        logger.debug("Starting skew detection analysis...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Enhanced edge detection for better line detection
        edges = self._enhance_edges_for_skew_detection(gray)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return SkewAnalysisResult(
                skew_angle=0.0,
                confidence=0.0,
                evidence="No lines detected for skew analysis",
                line_count=0,
                detection_method="hough_lines"
            )
        
        # Calculate skew angle with improved filtering
        skew_angle, confidence, line_count = self._calculate_skew_angle(lines)
        
        evidence = f"Skew angle {skew_angle:.1f}° detected from {line_count} lines"
        
        return SkewAnalysisResult(
            skew_angle=skew_angle,
            confidence=confidence,
            evidence=evidence,
            line_count=line_count,
            detection_method="hough_lines"
        )
    
    def _enhance_edges_for_skew_detection(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhance edges specifically for skew detection
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding to highlight text
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        
        # Dilate to connect text components
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Apply Canny edge detection
        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
        
        return edges
    
    def _calculate_skew_angle(self, lines: np.ndarray) -> Tuple[float, float, int]:
        """
        Calculate skew angle from detected lines
        
        Returns:
            Tuple[skew_angle, confidence, line_count]
        """
        # Extract angles from lines
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            # Only consider reasonable skew angles (typically < 45°)
            if abs(angle) < 45:
                angles.append(angle)
        
        if not angles:
            return 0.0, 0.0, 0
        
        # Use robust statistics to find skew angle
        skew_angle = np.median(angles)
        
        # Calculate confidence based on angle consistency
        angle_std = np.std(angles)
        confidence = max(0.0, min(1.0, 1.0 - (angle_std / 10.0)))
        
        # Boost confidence for clear skew
        if abs(skew_angle) > 1.0:
            confidence += 0.1
        
        confidence = min(confidence, 1.0)
        
        return float(skew_angle), confidence, len(angles)


# =============================================================================
# QUALITY ASSESSOR
# =============================================================================

class QualityAssessor(BaseDocumentAnalyzer):
    """
    Comprehensive image quality assessment
    """
    
    def analyze(self, image: np.ndarray) -> QualityAnalysisResult:
        """
        Assess image quality using various metrics
        
        Args:
            image: Input image as numpy array
            
        Returns:
            QualityAnalysisResult: Comprehensive quality assessment
        """
        self._validate_image(image)
        
        logger.debug("Starting quality assessment analysis...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Perform various quality assessments
        contrast_score = self._assess_contrast(gray)
        brightness_score = self._assess_brightness(gray)
        sharpness_score = self._assess_sharpness(gray)
        noise_level = self._assess_noise_level(gray)
        
        # Determine issues and recommendations
        detected_issues = []
        recommendations = []
        
        # Analyze results and generate recommendations
        if contrast_score < 0.4:
            detected_issues.append("low_contrast")
            recommendations.append(ProcessingAction.ENHANCE_CONTRAST)
        
        if brightness_score < 0.3:
            detected_issues.append("dark_image")
            recommendations.append(ProcessingAction.GAMMA_CORRECTION)
        
        if sharpness_score < 0.3:
            detected_issues.append("blurry_image")
            recommendations.append(ProcessingAction.SHARPEN)
        
        if noise_level > 0.6:
            detected_issues.append("noisy_image")
            recommendations.append(ProcessingAction.DENOISE)
        
        # Calculate overall quality score
        overall_score = (
            contrast_score * 0.3 +
            brightness_score * 0.2 +
            sharpness_score * 0.3 +
            (1 - noise_level) * 0.2
        )
        
        return QualityAnalysisResult(
            overall_score=overall_score,
            contrast_score=contrast_score,
            brightness_score=brightness_score,
            sharpness_score=sharpness_score,
            noise_level=noise_level,
            detected_issues=detected_issues,
            recommendations=recommendations
        )
    
    def _assess_contrast(self, gray: np.ndarray) -> float:
        """
        Assess image contrast using standard deviation
        """
        contrast = np.std(gray)
        # Normalize to 0-1 range (typical std for good documents: 40-80)
        normalized_contrast = min(contrast / 80.0, 1.0)
        return normalized_contrast
    
    def _assess_brightness(self, gray: np.ndarray) -> float:
        """
        Assess image brightness
        """
        brightness = np.mean(gray)
        # Normalize to 0-1 range (ideal brightness: 120-200)
        if brightness < 120:
            normalized_brightness = brightness / 120.0
        elif brightness > 200:
            normalized_brightness = max(0.0, 1.0 - (brightness - 200) / 55.0)
        else:
            normalized_brightness = 1.0
        
        return normalized_brightness
    
    def _assess_sharpness(self, gray: np.ndarray) -> float:
        """
        Assess image sharpness using Laplacian variance
        """
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range (typical variance for sharp images: 200-2000)
        normalized_sharpness = min(laplacian_var / 1000.0, 1.0)
        return normalized_sharpness
    
    def _assess_noise_level(self, gray: np.ndarray) -> float:
        """
        Assess noise level using local variance analysis
        """
        # Apply median filter and compare with original
        median_filtered = cv2.medianBlur(gray, 5)
        noise_estimate = np.mean(np.abs(gray.astype(float) - median_filtered.astype(float)))
        
        # Normalize to 0-1 range
        normalized_noise = min(noise_estimate / 20.0, 1.0)
        return normalized_noise


# =============================================================================
# CONTENT ANALYZER
# =============================================================================

class ContentAnalyzer(BaseDocumentAnalyzer):
    """
    Analyze document content characteristics
    """
    
    def analyze(self, image: np.ndarray) -> ContentAnalysisResult:
        """
        Analyze content characteristics like handwriting, columns, etc.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            ContentAnalysisResult: Content analysis results
        """
        self._validate_image(image)
        
        logger.debug("Starting content analysis...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Analyze handwriting
        has_handwriting, handwriting_confidence = self._detect_handwriting(gray)
        
        # Analyze column layout
        column_count = self._detect_columns(gray)
        
        # Determine layout complexity
        layout_complexity = self._assess_layout_complexity(gray)
        
        # Collect detected features
        detected_features = []
        if has_handwriting:
            detected_features.append("handwriting")
        if column_count > 1:
            detected_features.append("multi_column")
        if layout_complexity == "complex":
            detected_features.append("complex_layout")
        
        return ContentAnalysisResult(
            has_handwriting=has_handwriting,
            handwriting_confidence=handwriting_confidence,
            column_count=column_count,
            layout_complexity=layout_complexity,
            detected_features=detected_features
        )
    
    def _detect_handwriting(self, gray: np.ndarray) -> Tuple[bool, float]:
        """
        Detect handwriting using stroke width and regularity analysis
        """
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Analyze stroke characteristics
        stroke_irregularity = 0.0
        stroke_width_variation = 0.0
        valid_contours = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 5000:  # Reasonable character size
                # Calculate contour irregularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Circularity (4πA/P²) - lower for irregular shapes
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    stroke_irregularity += (1 - circularity)
                    
                    # Calculate stroke width variation
                    rect = cv2.boundingRect(contour)
                    aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 0
                    if aspect_ratio > 0:
                        stroke_width_variation += min(aspect_ratio, 1/aspect_ratio)
                    
                    valid_contours += 1
        
        if valid_contours == 0:
            return False, 0.0
        
        # Calculate averages
        avg_irregularity = stroke_irregularity / valid_contours
        avg_width_variation = stroke_width_variation / valid_contours
        
        # Combine metrics for handwriting detection
        handwriting_score = (avg_irregularity * 0.7 + (1 - avg_width_variation) * 0.3)
        handwriting_score = max(0.0, min(1.0, handwriting_score))
        
        # Threshold for handwriting detection
        has_handwriting = handwriting_score > 0.3
        
        return has_handwriting, handwriting_score
    
    def _detect_columns(self, gray: np.ndarray) -> int:
        """
        Detect number of columns using vertical projection
        """
        # Create vertical projection
        vertical_proj = np.sum(gray < 128, axis=0)
        
        # Smooth the projection
        kernel_size = max(5, min(20, len(vertical_proj) // 50))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(vertical_proj, kernel, mode='same')
        
        # Find valleys (column separators)
        threshold = np.mean(smoothed) * 0.3
        valleys = smoothed < threshold
        
        # Count distinct column regions
        column_regions = []
        in_valley = False
        start_col = 0
        min_column_width = len(vertical_proj) // 10  # Minimum column width
        
        for i, is_valley in enumerate(valleys):
            if not in_valley and is_valley:
                if i - start_col > min_column_width:
                    column_regions.append((start_col, i))
                in_valley = True
            elif in_valley and not is_valley:
                start_col = i
                in_valley = False
        
        # Add last column if exists
        if not in_valley and len(vertical_proj) - start_col > min_column_width:
            column_regions.append((start_col, len(vertical_proj)))
        
        return max(1, len(column_regions))
    
    def _assess_layout_complexity(self, gray: np.ndarray) -> str:
        """
        Assess layout complexity based on various features
        """
        # Find contours for layout analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "simple"
        
        # Calculate layout metrics
        num_components = len(contours)
        areas = [cv2.contourArea(c) for c in contours]
        
        # Filter out noise (very small components)
        significant_areas = [a for a in areas if a > 100]
        
        if not significant_areas:
            return "simple"
        
        # Calculate area variation
        area_std = np.std(significant_areas)
        area_mean = np.mean(significant_areas)
        area_cv = area_std / area_mean if area_mean > 0 else 0
        
        # Determine complexity based on metrics
        if num_components < 20 and area_cv < 1.0:
            return "simple"
        elif num_components < 100 and area_cv < 2.0:
            return "moderate"
        else:
            return "complex"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_condition_detection(condition: DocumentCondition, 
                              severity: SeverityLevel,
                              confidence: float,
                              evidence: str,
                              recommended_actions: List[ProcessingAction],
                              **kwargs) -> ConditionDetection:
    """
    Helper function to create ConditionDetection objects
    """
    return ConditionDetection(
        condition=condition,
        severity=severity,
        confidence=confidence,
        evidence=evidence,
        recommended_actions=recommended_actions,
        **kwargs
    )


def convert_rotation_result_to_condition(result: RotationAnalysisResult) -> Optional[ConditionDetection]:
    """
    Convert rotation analysis result to condition detection
    """
    if result.rotation_angle == 0 or result.confidence < 0.5:
        return None
    
    # Map rotation angle to condition and action
    condition_map = {
        90: (DocumentCondition.ROTATED_90_CCW, ProcessingAction.ROTATE_90_CW),
        180: (DocumentCondition.ROTATED_180, ProcessingAction.ROTATE_180),
        270: (DocumentCondition.ROTATED_90_CW, ProcessingAction.ROTATE_90_CCW)
    }
    
    if result.rotation_angle in condition_map:
        condition, action = condition_map[result.rotation_angle]
        severity = SeverityLevel.SEVERE if result.rotation_angle in [90, 270] else SeverityLevel.MODERATE
        
        return create_condition_detection(
            condition=condition,
            severity=severity,
            confidence=result.confidence,
            evidence=result.evidence,
            recommended_actions=[action],
            orientation_indicators=[
                f"Detected rotation: {result.rotation_angle}°",
                f"Text orientation score: {result.text_orientation_score:.2f}",
                f"Edge orientation score: {result.edge_orientation_score:.2f}"
            ],
            detection_reasoning=f"Combined CV analysis indicates {result.rotation_angle}° rotation needed"
        )
    
    return None


def convert_skew_result_to_condition(result: SkewAnalysisResult) -> Optional[ConditionDetection]:
    """
    Convert skew analysis result to condition detection
    """
    if abs(result.skew_angle) < 2.0 or result.confidence < 0.5:
        return None
    
    severity = SeverityLevel.MODERATE if abs(result.skew_angle) < 10 else SeverityLevel.SEVERE
    
    return create_condition_detection(
        condition=DocumentCondition.SKEWED,
        severity=severity,
        confidence=result.confidence,
        evidence=result.evidence,
        recommended_actions=[ProcessingAction.DESKEW],
        detection_reasoning=f"Hough line analysis detected {result.skew_angle:.1f}° skew"
    )


def convert_quality_result_to_conditions(result: QualityAnalysisResult) -> List[ConditionDetection]:
    """
    Convert quality analysis result to condition detections
    """
    conditions = []
    
    # Map quality issues to conditions
    issue_map = {
        "low_contrast": DocumentCondition.LOW_CONTRAST,
        "dark_image": DocumentCondition.DARK_BACKGROUND,
        "blurry_image": DocumentCondition.BLURRY,
        "noisy_image": DocumentCondition.POOR_QUALITY
    }
    
    for issue in result.detected_issues:
        if issue in issue_map:
            condition = issue_map[issue]
            
            # Determine severity based on scores
            if issue == "low_contrast":
                severity = SeverityLevel.SEVERE if result.contrast_score < 0.2 else SeverityLevel.MODERATE
                evidence = f"Low contrast detected (score: {result.contrast_score:.2f})"
            elif issue == "dark_image":
                severity = SeverityLevel.MODERATE
                evidence = f"Dark background detected (score: {result.brightness_score:.2f})"
            elif issue == "blurry_image":
                severity = SeverityLevel.MODERATE
                evidence = f"Blur detected (sharpness score: {result.sharpness_score:.2f})"
            elif issue == "noisy_image":
                severity = SeverityLevel.MODERATE
                evidence = f"High noise level detected (level: {result.noise_level:.2f})"
            else:
                severity = SeverityLevel.MODERATE
                evidence = f"Quality issue: {issue}"
            
            # Get corresponding recommendations
            issue_actions = [action for action in result.recommendations 
                           if _action_addresses_issue(action, issue)]
            
            conditions.append(create_condition_detection(
                condition=condition,
                severity=severity,
                confidence=0.8,  # Quality assessment generally reliable
                evidence=evidence,
                recommended_actions=issue_actions or [ProcessingAction.ENHANCE_CONTRAST]
            ))
    
    return conditions


def convert_content_result_to_conditions(result: ContentAnalysisResult) -> List[ConditionDetection]:
    """
    Convert content analysis result to condition detections
    """
    conditions = []
    
    # Handwriting detection
    if result.has_handwriting:
        conditions.append(create_condition_detection(
            condition=DocumentCondition.HANDWRITTEN,
            severity=SeverityLevel.MODERATE,
            confidence=result.handwriting_confidence,
            evidence=f"Handwriting detected (confidence: {result.handwriting_confidence:.2f})",
            recommended_actions=[ProcessingAction.ENHANCE_CONTRAST, ProcessingAction.DENOISE]
        ))
    
    # Multi-column detection
    if result.column_count > 1:
        conditions.append(create_condition_detection(
            condition=DocumentCondition.MULTI_COLUMN,
            severity=SeverityLevel.MILD,
            confidence=0.8,
            evidence=f"Multi-column layout detected ({result.column_count} columns)",
            recommended_actions=[ProcessingAction.NO_ACTION]
        ))
    
    # Complex layout detection
    if result.layout_complexity == "complex":
        conditions.append(create_condition_detection(
            condition=DocumentCondition.COMPLEX_LAYOUT,
            severity=SeverityLevel.MILD,
            confidence=0.7,
            evidence=f"Complex layout detected",
            recommended_actions=[ProcessingAction.NO_ACTION]
        ))
    
    return conditions


def _action_addresses_issue(action: ProcessingAction, issue: str) -> bool:
    """Helper function to check if an action addresses a specific issue"""
    action_issue_map = {
        ProcessingAction.ENHANCE_CONTRAST: ["low_contrast"],
        ProcessingAction.GAMMA_CORRECTION: ["dark_image"],
        ProcessingAction.SHARPEN: ["blurry_image"],
        ProcessingAction.DENOISE: ["noisy_image"]
    }
    
    return issue in action_issue_map.get(action, [])