"""
Modular Preprocessing System Usage Example
Demonstrates the refactored document preprocessing system with modular analyzers
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the refactored components
from src.analyzers.document_condition_cv_analyzer import (
    DocumentPreprocessor, DocumentConditionAnalyzer, main_preprocessing_workflow
)
from src.utils.document_analysis_utils import (
    RotationDetector, SkewDetector, QualityAssessor, ContentAnalyzer
)
from src.pydantic_models.document_condition_models import (
    DocumentCondition, ProcessingAction, SeverityLevel
)


# =============================================================================
# BASIC USAGE EXAMPLES
# =============================================================================

def basic_preprocessing_example():
    """
    Basic usage example with the refactored system
    """
    print("=== Basic Preprocessing Example (Refactored) ===")
    
    pdf_path = "data/inputs/var3.pdf"
    output_dir = "data/output/refactored_preprocessing"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please provide a valid PDF file for testing.")
        return
    
    try:
        # Process with the refactored system
        result = main_preprocessing_workflow(
            pdf_path=pdf_path, 
            output_dir=output_dir,
            enable_debug=True  # Enable detailed logging
        )
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Output directory: {result.processed_path}")
        print(f"📊 Quality score: {result.final_quality_score:.2f}")
        print(f"⏱️ Processing time: {result.total_processing_time:.2f}s")
        
        # Show detailed analysis for each page
        print(f"\n📋 Detailed Analysis Results:")
        for i, page_result in enumerate(result.processing_results, 1):
            print(f"\n  Page {i}:")
            print(f"    Actions applied: {[a.value for a in page_result.actions_applied]}")
            print(f"    Overall quality: {page_result.metadata.get('overall_quality', 'N/A')}")
            print(f"    Processing priority: {page_result.metadata.get('processing_priority', 'N/A')}")
            
            # Show conditions detected
            assessment = page_result.metadata.get('condition_assessment')
            if assessment:
                primary_issues = assessment.get('primary_issues', [])
                if primary_issues:
                    print(f"    Primary issues: {[issue['condition'] for issue in primary_issues]}")
        
        return result
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        return None


def individual_analyzer_example():
    """
    Example of using individual analyzers separately
    """
    print("\n=== Individual Analyzer Example ===")
    
    image_path = "samples/page_3.png"
    
    if not os.path.exists(image_path):
        print(f"Individual analyzer example requires image at: {image_path}")
        return
    
    # Load image
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    try:
        # Initialize individual analyzers
        rotation_detector = RotationDetector(confidence_threshold=0.7)
        skew_detector = SkewDetector(confidence_threshold=0.7)
        quality_assessor = QualityAssessor(confidence_threshold=0.7)
        content_analyzer = ContentAnalyzer(confidence_threshold=0.7)
        
        # Run individual analyses
        print(f"\n🔄 Rotation Analysis:")
        rotation_result = rotation_detector.analyze(image)
        print(f"  Rotation angle: {rotation_result.rotation_angle}°")
        print(f"  Confidence: {rotation_result.confidence:.2f}")
        print(f"  Evidence: {rotation_result.evidence}")
        print(f"  Text orientation: {rotation_result.text_orientation_score:.2f}")
        print(f"  Edge orientation: {rotation_result.edge_orientation_score:.2f}")
        
        print(f"\n📐 Skew Analysis:")
        skew_result = skew_detector.analyze(image)
        print(f"  Skew angle: {skew_result.skew_angle:.2f}°")
        print(f"  Confidence: {skew_result.confidence:.2f}")
        print(f"  Evidence: {skew_result.evidence}")
        print(f"  Lines analyzed: {skew_result.line_count}")
        
        print(f"\n📊 Quality Assessment:")
        quality_result = quality_assessor.analyze(image)
        print(f"  Overall score: {quality_result.overall_score:.2f}")
        print(f"  Contrast: {quality_result.contrast_score:.2f}")
        print(f"  Brightness: {quality_result.brightness_score:.2f}")
        print(f"  Sharpness: {quality_result.sharpness_score:.2f}")
        print(f"  Noise level: {quality_result.noise_level:.2f}")
        print(f"  Issues detected: {quality_result.detected_issues}")
        
        print(f"\n📄 Content Analysis:")
        content_result = content_analyzer.analyze(image)
        print(f"  Has handwriting: {content_result.has_handwriting}")
        print(f"  Handwriting confidence: {content_result.handwriting_confidence:.2f}")
        print(f"  Column count: {content_result.column_count}")
        print(f"  Layout complexity: {content_result.layout_complexity}")
        print(f"  Detected features: {content_result.detected_features}")
        
        return {
            'rotation': rotation_result,
            'skew': skew_result,
            'quality': quality_result,
            'content': content_result
        }
        
    except Exception as e:
        print(f"❌ Individual analysis failed: {str(e)}")
        return None


def condition_analyzer_example():
    """
    Example of using the refactored DocumentConditionAnalyzer
    """
    print("\n=== Refactored Document Condition Analyzer Example ===")
    
    image_path = "data/input/sample_page.png"
    
    if not os.path.exists(image_path):
        print(f"Condition analyzer example requires image at: {image_path}")
        return
    
    try:
        # Initialize the refactored analyzer
        analyzer = DocumentConditionAnalyzer(
            confidence_threshold=0.7,
            enable_debug=True
        )
        
        # Perform comprehensive analysis
        assessment = analyzer.analyze_document_condition(image_path)
        
        print(f"\n📋 Comprehensive Analysis Results:")
        print(f"Overall quality: {assessment.overall_quality}")
        print(f"Processing priority: {assessment.processing_priority}")
        print(f"Success rate estimate: {assessment.estimated_success_rate:.2f}")
        
        # Show primary issues
        if assessment.primary_issues:
            print(f"\n🚨 Primary Issues:")
            for issue in assessment.primary_issues:
                print(f"  - {issue.condition.value} (severity: {issue.severity.value})")
                print(f"    Confidence: {issue.confidence:.2f}")
                print(f"    Evidence: {issue.evidence}")
                print(f"    Recommended actions: {[a.value for a in issue.recommended_actions]}")
        
        # Show secondary issues
        if assessment.secondary_issues:
            print(f"\n⚠️  Secondary Issues:")
            for issue in assessment.secondary_issues:
                print(f"  - {issue.condition.value} (severity: {issue.severity.value})")
        
        # Show processing recommendations
        print(f"\n🔧 Processing Recommendations:")
        for action in assessment.processing_recommendations:
            print(f"  - {action.value}")
        
        # Show special handling notes
        if assessment.special_handling_notes:
            print(f"\n📝 Special Notes: {assessment.special_handling_notes}")
        
        # Show detailed analysis report (if debug enabled)
        if hasattr(analyzer, 'get_detailed_analysis_report'):
            detailed_report = analyzer.get_detailed_analysis_report()
            print(f"\n🔍 Detailed Analysis Report:")
            print(f"  Image dimensions: {detailed_report['image_info']['width']}x{detailed_report['image_info']['height']}")
            print(f"  Aspect ratio: {detailed_report['image_info']['aspect_ratio']:.2f}")
            
            # Show individual analysis results
            if 'rotation' in detailed_report['analysis_results']:
                rotation_data = detailed_report['analysis_results']['rotation']
                print(f"  Rotation analysis: {rotation_data.get('rotation_angle', 'N/A')}° (confidence: {rotation_data.get('confidence', 'N/A')})")
            
            if 'quality' in detailed_report['analysis_results']:
                quality_data = detailed_report['analysis_results']['quality']
                print(f"  Quality score: {quality_data.get('overall_score', 'N/A')}")
        
        return assessment
        
    except Exception as e:
        print(f"❌ Condition analysis failed: {str(e)}")
        return None


def custom_processing_example():
    """
    Example of custom processing with specific analyzers
    """
    print("\n=== Custom Processing Example ===")
    
    pdf_path = "data/input/var3.pdf"
    output_dir = "data/output/custom_processing"
    
    if not os.path.exists(pdf_path):
        print(f"Custom processing example requires PDF at: {pdf_path}")
        return
    
    try:
        # Initialize preprocessor with custom settings
        preprocessor = DocumentPreprocessor(
            output_dir=output_dir,
            enable_debug=True
        )
        
        # Customize analyzer settings
        preprocessor.condition_analyzer.confidence_threshold = 0.6  # Lower threshold
        preprocessor.condition_analyzer.rotation_detector.set_debug_mode(True)
        
        # Process document
        result = preprocessor.process_pdf(pdf_path, document_id="custom_test")
        
        print(f"\n✅ Custom processing completed!")
        print(f"📁 Output: {result.processed_path}")
        print(f"🎯 Success: {result.success}")
        
        # Show processing statistics
        rotation_corrections = 0
        quality_enhancements = 0
        
        for page_result in result.processing_results:
            for action in page_result.actions_applied:
                if action in [ProcessingAction.ROTATE_90_CW, ProcessingAction.ROTATE_90_CCW, ProcessingAction.ROTATE_180]:
                    rotation_corrections += 1
                elif action in [ProcessingAction.ENHANCE_CONTRAST, ProcessingAction.DENOISE, ProcessingAction.SHARPEN]:
                    quality_enhancements += 1
        
        print(f"\n📊 Processing Statistics:")
        print(f"  Rotation corrections: {rotation_corrections}")
        print(f"  Quality enhancements: {quality_enhancements}")
        print(f"  Total pages: {len(result.processing_results)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Custom processing failed: {str(e)}")
        return None
    
    finally:
        # Clean up
        if 'preprocessor' in locals():
            preprocessor.cleanup()


def performance_comparison():
    """
    Compare performance between original and refactored system
    """
    print("\n=== Performance Comparison ===")
    
    image_path = "data/input/sample_page.png"
    
    if not os.path.exists(image_path):
        print(f"Performance comparison requires image at: {image_path}")
        return
    
    import time
    
    try:
        # Load image
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        # Test individual analyzers
        print(f"\n⏱️  Individual Analyzer Performance:")
        
        # Rotation detection
        start_time = time.time()
        rotation_detector = RotationDetector()
        rotation_result = rotation_detector.analyze(image)
        rotation_time = time.time() - start_time
        print(f"  Rotation detection: {rotation_time:.3f}s")
        
        # Skew detection
        start_time = time.time()
        skew_detector = SkewDetector()
        skew_result = skew_detector.analyze(image)
        skew_time = time.time() - start_time
        print(f"  Skew detection: {skew_time:.3f}s")
        
        # Quality assessment
        start_time = time.time()
        quality_assessor = QualityAssessor()
        quality_result = quality_assessor.analyze(image)
        quality_time = time.time() - start_time
        print(f"  Quality assessment: {quality_time:.3f}s")
        
        # Content analysis
        start_time = time.time()
        content_analyzer = ContentAnalyzer()
        content_result = content_analyzer.analyze(image)
        content_time = time.time() - start_time
        print(f"  Content analysis: {content_time:.3f}s")
        
        # Combined analysis
        start_time = time.time()
        analyzer = DocumentConditionAnalyzer()
        assessment = analyzer.analyze_document_condition(image_path)
        combined_time = time.time() - start_time
        print(f"  Combined analysis: {combined_time:.3f}s")
        
        # Show efficiency
        individual_total = rotation_time + skew_time + quality_time + content_time
        efficiency = individual_total / combined_time
        print(f"\n📊 Efficiency Analysis:")
        print(f"  Individual total: {individual_total:.3f}s")
        print(f"  Combined total: {combined_time:.3f}s")
        print(f"  Efficiency ratio: {efficiency:.2f}x")
        
        return {
            'rotation_time': rotation_time,
            'skew_time': skew_time,
            'quality_time': quality_time,
            'content_time': content_time,
            'combined_time': combined_time,
            'efficiency': efficiency
        }
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {str(e)}")
        return None


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def run_modular_examples():
    """
    Run all modular system examples
    """
    print("🚀 Starting Modular Document Preprocessing Demonstration")
    print("=" * 70)
    
    # Create necessary directories
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    examples = [
        ("Basic Preprocessing", basic_preprocessing_example),
        # ("Individual Analyzers", individual_analyzer_example),
        # ("Condition Analyzer", condition_analyzer_example),
        # ("Custom Processing", custom_processing_example),
        # ("Performance Comparison", performance_comparison)
    ]
    
    results = {}
    
    for name, example_func in examples:
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
        
        try:
            result = example_func()
            results[name] = result
            print(f"\n✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
            results[name] = None
        
        print(f"\n{'='*70}")
    
    # Summary
    print(f"\n🎉 All examples completed!")
    print(f"📊 Results Summary:")
    for name, result in results.items():
        status = "✅ Success" if result is not None else "❌ Failed"
        print(f"  {name}: {status}")
    
    print(f"\n📁 Check the data/output directory for results")
    print(f"📋 Log files contain detailed analysis information")


if __name__ == "__main__":
    run_modular_examples()