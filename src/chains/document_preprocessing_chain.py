"""
Simplified Document Preprocessing Chain using LangChain LCEL
Focus: PDF → Images → Condition Analysis → Image Preprocessing → Preprocessed Images

File: src/chains/simple_preprocessing_chain.py
"""

import os
import tempfile
import logging
import shutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from langchain_core.runnables import (
    Runnable, 
    RunnableLambda, 
    RunnablePassthrough,
    RunnableParallel
)
from langchain_core.runnables.utils import Input, Output

from ..analyzers.document_condition_analyzer import DocumentConditionAnalyzer
from ..utils.image_utils import ImageProcessor, convert_pdf_to_images
from ..pydantic_models.document_condition_models import (
    DocumentConditionAssessment,
    ProcessingAction,
    SeverityLevel
)

logger = logging.getLogger(__name__)


# =============================================================================
# APPROACH 1: Using RunnableLambda (Simpler, More Concise)
# =============================================================================

class SimpleDocumentPreprocessor:
    """Simple document preprocessing chain using RunnableLambda"""
    
    def __init__(self, openai_api_key: str, temp_dir: Optional[str] = None, output_dir: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.output_dir = output_dir or "data/outputs/preprocessed"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.condition_analyzer = DocumentConditionAnalyzer(openai_api_key)
        self.image_processor = ImageProcessor(self.output_dir, enable_advanced=True)  # Use output_dir instead of temp_dir
        
        # Build the chain using RunnableLambda
        self.chain = self._build_simple_chain()
    
    def _build_simple_chain(self) -> Runnable:
        """Build preprocessing chain using RunnableLambda functions"""
        
        def convert_pdf_step(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Convert PDF to images or handle single image"""
            file_path = input_data['file_path']
            output_dir = input_data.get('output_dir', self.output_dir)
            
            # Create document-specific subdirectory
            doc_name = Path(file_path).stem
            doc_output_dir = os.path.join(output_dir, doc_name)
            os.makedirs(doc_output_dir, exist_ok=True)
            
            if file_path.lower().endswith('.pdf'):
                logger.info(f"Converting PDF to images: {file_path}")
                images = convert_pdf_to_images(file_path, dpi=300)
                
                if not images:
                    raise ValueError("Failed to convert PDF to images")
                
                # Save images with meaningful names
                image_paths = []
                for i, img in enumerate(images):
                    image_path = os.path.join(doc_output_dir, f"{doc_name}_page_{i+1:03d}_original.png")
                    img.save(image_path, 'PNG')
                    image_paths.append(image_path)
                    logger.info(f"Saved page {i+1} to: {image_path}")
                
                return {
                    **input_data,
                    'image_paths': image_paths,
                    'doc_output_dir': doc_output_dir,
                    'document_name': doc_name,
                    'is_pdf': True,
                    'total_pages': len(image_paths)
                }
            else:
                # Single image file - copy to output directory with consistent naming
                original_name = Path(file_path).stem
                copied_path = os.path.join(doc_output_dir, f"{original_name}_original.png")
                
                # Copy image to output directory
                import shutil
                if file_path != copied_path:
                    shutil.copy2(file_path, copied_path)
                    logger.info(f"Copied image to: {copied_path}")
                
                return {
                    **input_data,
                    'image_paths': [copied_path],
                    'doc_output_dir': doc_output_dir,
                    'document_name': original_name,
                    'is_pdf': False,
                    'total_pages': 1
                }
        
        def analyze_conditions_step(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze document conditions for each image"""
            image_paths = input_data['image_paths']
            assessments = []
            
            logger.info(f"Analyzing conditions for {len(image_paths)} image(s)")
            
            for i, image_path in enumerate(image_paths):
                try:
                    assessment = self.condition_analyzer.analyze_document_condition(image_path)
                    assessments.append(assessment)
                    logger.info(f"Page {i+1}: {len(assessment.primary_issues)} issues detected")
                except Exception as e:
                    logger.error(f"Failed to analyze page {i+1}: {e}")
                    # Create fallback assessment
                    fallback = self.condition_analyzer._create_enhanced_fallback_assessment(image_path)
                    assessments.append(fallback)
            
            return {
                **input_data,
                'condition_assessments': assessments
            }
        
        def preprocess_images_step(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Apply preprocessing based on condition analysis - FIXED VERSION"""
            image_paths = input_data['image_paths']
            assessments_data = input_data['condition_assessments']
            doc_output_dir = input_data['doc_output_dir']
            document_name = input_data['document_name']
            
            processed_paths = []
            processing_results = []
            
            logger.info(f"Preprocessing {len(image_paths)} image(s)")
            
            for i, (image_path, assessment_data) in enumerate(zip(image_paths, assessments_data)):
                # Re-validate the assessment data to ensure correct types (Enums).
                if isinstance(assessment_data, dict):
                    try:
                        assessment = DocumentConditionAssessment.model_validate(assessment_data)
                    except Exception as validation_error:
                        logger.error(f"Failed to validate assessment data for page {i+1}: {validation_error}")
                        # Fallback: copy original image if assessment data is invalid
                        processed_filename = f"{document_name}_page_{i+1:03d}_preprocessed.png" if input_data['is_pdf'] else f"{document_name}_preprocessed.png"
                        processed_path = os.path.join(doc_output_dir, processed_filename)
                        shutil.copy2(image_path, processed_path)
                        processed_paths.append(processed_path)
                        processing_results.append(None)
                        continue
                else:
                    assessment = assessment_data

                try:
                    # Generate meaningful output filename
                    if input_data['is_pdf']:
                        processed_filename = f"{document_name}_page_{i+1:03d}_preprocessed.png"
                    else:
                        processed_filename = f"{document_name}_preprocessed.png"
                    
                    processed_path = os.path.join(doc_output_dir, processed_filename)
                    
                    # CRITICAL FIX: Use the FINAL processing_recommendations from the assessment
                    # This ensures we use the corrected recommendations after validation
                    final_recommendations = assessment.processing_recommendations
                    
                    logger.info(f"Page {i+1}: Final processing recommendations: {[action.value for action in final_recommendations]}")
                    
                    # Determine if preprocessing is needed
                    if (not final_recommendations or 
                        assessment.processing_priority > 3 or
                        assessment.estimated_success_rate > 0.8):
                        
                        logger.info(f"Page {i+1}: No preprocessing needed, using original")
                        shutil.copy2(image_path, processed_path)
                        processed_paths.append(processed_path)
                        processing_results.append(None)
                        continue
                    
                    # Apply preprocessing using FINAL recommendations
                    logger.info(f"Page {i+1}: Applying {len(final_recommendations)} actions")
                    logger.info(f"Actions: {[action.value for action in final_recommendations]}")
                    
                    result = self.image_processor.preprocess_image(
                        image_path=image_path,
                        processing_level="custom",
                        custom_actions=final_recommendations  # Use final recommendations here!
                    )
                    
                    if hasattr(result, 'output_path'):
                        shutil.move(result.output_path, processed_path)
                        processed_paths.append(processed_path)
                        processing_results.append(result)
                        logger.info(f"Saved preprocessed image to: {processed_path}")
                    else:
                        shutil.move(result, processed_path)
                        processed_paths.append(processed_path)
                        processing_results.append(None)
                        logger.info(f"Saved preprocessed image to: {processed_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to preprocess page {i+1}: {e}")
                    # Fallback: copy original image on processing failure
                    processed_filename = f"{document_name}_page_{i+1:03d}_preprocessed.png" if input_data['is_pdf'] else f"{document_name}_preprocessed.png"
                    processed_path = os.path.join(doc_output_dir, processed_filename)
                    shutil.copy2(image_path, processed_path)
                    processed_paths.append(processed_path)
                    processing_results.append(None)
            
            return {
                **input_data,
                'processed_image_paths': processed_paths,
                'processing_results': processing_results
            }
        
        def compile_results_step(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Compile final results"""
            processing_time = datetime.now().timestamp() - input_data.get('start_time', datetime.now().timestamp())
            
            successful_preprocessing = sum(1 for result in input_data['processing_results'] if result is not None)
            
            logger.info(f"Preprocessing completed:")
            logger.info(f"  - Total pages: {input_data['total_pages']}")
            logger.info(f"  - Pages preprocessed: {successful_preprocessing}")
            logger.info(f"  - Output directory: {input_data['doc_output_dir']}")
            logger.info(f"  - Processing time: {processing_time:.2f}s")
            
            return {
                'original_file_path': input_data['file_path'],
                'output_directory': input_data['doc_output_dir'],
                'document_name': input_data['document_name'],
                'is_pdf': input_data['is_pdf'],
                'total_pages': input_data['total_pages'],
                'original_image_paths': input_data['image_paths'],
                'processed_image_paths': input_data['processed_image_paths'],
                'condition_assessments': input_data['condition_assessments'],
                'processing_results': input_data['processing_results'],
                'pages_preprocessed': successful_preprocessing,
                'pages_skipped': input_data['total_pages'] - successful_preprocessing,
                'success': True,
                'processing_time': processing_time
            }
        
        chain = (
            RunnableLambda(lambda x: {**x, 'start_time': datetime.now().timestamp()})
            | RunnableLambda(convert_pdf_step)
            | RunnableLambda(analyze_conditions_step) 
            | RunnableLambda(preprocess_images_step)
            | RunnableLambda(compile_results_step)
        )
        
        return chain
    
    def process(self, file_path: str, output_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Process a PDF or image file through the preprocessing pipeline"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        actual_output_dir = output_dir or self.output_dir
        os.makedirs(actual_output_dir, exist_ok=True)
        
        input_data = {
            'file_path': file_path,
            'output_dir': actual_output_dir,
            **kwargs
        }
        
        logger.info(f"Starting document preprocessing: {file_path}")
        logger.info(f"Output directory: {actual_output_dir}")
        
        result = self.chain.invoke(input_data)
        
        logger.info(f"✅ Preprocessing completed in {result['processing_time']:.2f}s")
        logger.info(f"📁 Results saved to: {result['output_directory']}")
        
        return result


# =============================================================================
# APPROACH 2: Using Separate Runnable Classes (More Structured)
# =============================================================================

class PDFConverter(Runnable):
    """Convert PDF to images or handle single image"""
    
    def __init__(self, temp_dir: str, output_dir: str):
        self.temp_dir = temp_dir
        self.output_dir = output_dir
    
    def invoke(self, input_data: Dict[str, Any], config=None) -> Dict[str, Any]:
        file_path = input_data['file_path']
        output_dir = input_data.get('output_dir', self.output_dir)
        
        doc_name = Path(file_path).stem
        doc_output_dir = os.path.join(output_dir, doc_name)
        os.makedirs(doc_output_dir, exist_ok=True)
        
        if file_path.lower().endswith('.pdf'):
            logger.info(f"Converting PDF to images: {file_path}")
            images = convert_pdf_to_images(file_path, dpi=300)
            
            if not images:
                raise ValueError("Failed to convert PDF to images")
            
            image_paths = []
            for i, img in enumerate(images):
                image_path = os.path.join(doc_output_dir, f"{doc_name}_page_{i+1:03d}_original.png")
                img.save(image_path, 'PNG')
                image_paths.append(image_path)
                logger.info(f"Saved page {i+1} to: {image_path}")
            
            return {
                **input_data,
                'image_paths': image_paths,
                'doc_output_dir': doc_output_dir,
                'document_name': doc_name,
                'is_pdf': True,
                'total_pages': len(image_paths)
            }
        else:
            original_name = Path(file_path).stem
            copied_path = os.path.join(doc_output_dir, f"{original_name}_original.png")
            
            if file_path != copied_path:
                shutil.copy2(file_path, copied_path)
                logger.info(f"Copied image to: {copied_path}")
            
            return {
                **input_data,
                'image_paths': [copied_path],
                'doc_output_dir': doc_output_dir,
                'document_name': original_name,
                'is_pdf': False,
                'total_pages': 1
            }


class ConditionAnalyzer(Runnable):
    """Analyze document conditions for multiple images"""
    
    def __init__(self, analyzer: DocumentConditionAnalyzer):
        self.analyzer = analyzer
    
    def invoke(self, input_data: Dict[str, Any], config=None) -> Dict[str, Any]:
        image_paths = input_data['image_paths']
        assessments = []
        
        logger.info(f"Analyzing conditions for {len(image_paths)} image(s)")
        
        for i, image_path in enumerate(image_paths):
            try:
                assessment = self.analyzer.analyze_document_condition(image_path)
                assessments.append(assessment)
                logger.info(f"Page {i+1}: {len(assessment.primary_issues)} issues detected")
            except Exception as e:
                logger.error(f"Failed to analyze page {i+1}: {e}")
                fallback = self.analyzer._create_enhanced_fallback_assessment(image_path)
                assessments.append(fallback)
        
        return {
            **input_data,
            'condition_assessments': assessments
        }


class ConditionalImageProcessor(Runnable):
    """Apply preprocessing based on condition analysis"""
    
    def __init__(self, image_processor: ImageProcessor):
        self.image_processor = image_processor
    
    def invoke(self, input_data: Dict[str, Any], config=None) -> Dict[str, Any]:
        image_paths = input_data['image_paths']
        assessments_data = input_data['condition_assessments']
        doc_output_dir = input_data['doc_output_dir']
        document_name = input_data['document_name']
        
        processed_paths = []
        processing_results = []
        
        logger.info(f"Preprocessing {len(image_paths)} image(s)")
        
        for i, (image_path, assessment_data) in enumerate(zip(image_paths, assessments_data)):
            # --- START OF FIX ---
            # Re-validate the assessment data to ensure correct types (Enums).
            # This handles cases where data might be serialized to dicts between chain steps.
            if isinstance(assessment_data, dict):
                try:
                    assessment = DocumentConditionAssessment.model_validate(assessment_data)
                except Exception as validation_error:
                    logger.error(f"Failed to validate assessment data for page {i+1}: {validation_error}")
                    # Fallback: copy original image if assessment data is invalid
                    processed_filename = f"{document_name}_page_{i+1:03d}_preprocessed.png" if input_data['is_pdf'] else f"{document_name}_preprocessed.png"
                    processed_path = os.path.join(doc_output_dir, processed_filename)
                    shutil.copy2(image_path, processed_path)
                    processed_paths.append(processed_path)
                    processing_results.append(None)
                    continue
            else:
                assessment = assessment_data
            # --- END OF FIX ---

            try:
                # Generate meaningful output filename
                if input_data['is_pdf']:
                    processed_filename = f"{document_name}_page_{i+1:03d}_preprocessed.png"
                else:
                    processed_filename = f"{document_name}_preprocessed.png"
                
                processed_path = os.path.join(doc_output_dir, processed_filename)
                
                # Determine if preprocessing is needed
                if (not assessment.processing_recommendations or 
                    assessment.processing_priority > 3 or
                    assessment.estimated_success_rate > 0.8):
                    
                    logger.info(f"Page {i+1}: No preprocessing needed, using original")
                    shutil.copy2(image_path, processed_path)
                    processed_paths.append(processed_path)
                    processing_results.append(None)
                    continue
                
                # Apply preprocessing
                logger.info(f"Page {i+1}: Applying {len(assessment.processing_recommendations)} actions")
                logger.info(f"Actions: {[action.value for action in assessment.processing_recommendations]}")
                
                result = self.image_processor.preprocess_image(
                    image_path=image_path,
                    processing_level="custom",
                    custom_actions=assessment.processing_recommendations
                )
                
                if hasattr(result, 'output_path'):
                    shutil.move(result.output_path, processed_path)
                    processed_paths.append(processed_path)
                    processing_results.append(result)
                    logger.info(f"Saved preprocessed image to: {processed_path}")
                else:
                    shutil.move(result, processed_path)
                    processed_paths.append(processed_path)
                    processing_results.append(None)
                    logger.info(f"Saved preprocessed image to: {processed_path}")
                
            except Exception as e:
                logger.error(f"Failed to preprocess page {i+1}: {e}")
                # Fallback: copy original image on processing failure
                processed_filename = f"{document_name}_page_{i+1:03d}_preprocessed.png" if input_data['is_pdf'] else f"{document_name}_preprocessed.png"
                processed_path = os.path.join(doc_output_dir, processed_filename)
                shutil.copy2(image_path, processed_path)
                processed_paths.append(processed_path)
                processing_results.append(None)
        
        return {
            **input_data,
            'processed_image_paths': processed_paths,
            'processing_results': processing_results
        }


class StructuredDocumentPreprocessor:
    """Document preprocessing chain using separate Runnable classes"""
    
    def __init__(self, openai_api_key: str, temp_dir: Optional[str] = None, output_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.output_dir = output_dir or "data/outputs/preprocessed"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        condition_analyzer = DocumentConditionAnalyzer(openai_api_key)
        image_processor = ImageProcessor(self.output_dir, enable_advanced=True)
        
        pdf_converter = PDFConverter(self.temp_dir, self.output_dir)
        condition_analyzer_runnable = ConditionAnalyzer(condition_analyzer)
        image_processor_runnable = ConditionalImageProcessor(image_processor)
        
        self.chain = (
            RunnableLambda(lambda x: {**x, 'start_time': datetime.now().timestamp()})
            | pdf_converter
            | condition_analyzer_runnable
            | image_processor_runnable
            | RunnableLambda(self._compile_results)
        )
    
    def _compile_results(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final results"""
        processing_time = datetime.now().timestamp() - input_data.get('start_time', datetime.now().timestamp())
        
        successful_preprocessing = sum(1 for result in input_data['processing_results'] if result is not None)
        
        logger.info(f"Preprocessing completed:")
        logger.info(f"  - Total pages: {input_data['total_pages']}")
        logger.info(f"  - Pages preprocessed: {successful_preprocessing}")
        logger.info(f"  - Output directory: {input_data['doc_output_dir']}")
        logger.info(f"  - Processing time: {processing_time:.2f}s")
        
        return {
            'original_file_path': input_data['file_path'],
            'output_directory': input_data['doc_output_dir'],
            'document_name': input_data['document_name'],
            'is_pdf': input_data['is_pdf'],
            'total_pages': input_data['total_pages'],
            'original_image_paths': input_data['image_paths'],
            'processed_image_paths': input_data['processed_image_paths'],
            'condition_assessments': input_data['condition_assessments'],
            'processing_results': input_data['processing_results'],
            'pages_preprocessed': successful_preprocessing,
            'pages_skipped': input_data['total_pages'] - successful_preprocessing,
            'success': True,
            'processing_time': processing_time
        }
    
    def process(self, file_path: str, output_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Process a PDF or image file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        actual_output_dir = output_dir or self.output_dir
        os.makedirs(actual_output_dir, exist_ok=True)
        
        input_data = {
            'file_path': file_path,
            'output_dir': actual_output_dir,
            **kwargs
        }
        
        logger.info(f"Starting document preprocessing: {file_path}")
        logger.info(f"Output directory: {actual_output_dir}")
        
        result = self.chain.invoke(input_data)
        
        logger.info(f"✅ Preprocessing completed in {result['processing_time']:.2f}s")
        logger.info(f"📁 Results saved to: {result['output_directory']}")
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS & EXAMPLES
# =============================================================================

def preprocess_document(file_path: str, 
                        openai_api_key: str,
                        output_dir: Optional[str] = None,
                        approach: str = "simple",
                        temp_dir: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for document preprocessing"""
    if approach == "simple":
        preprocessor = SimpleDocumentPreprocessor(openai_api_key, temp_dir, output_dir)
    elif approach == "structured":
        preprocessor = StructuredDocumentPreprocessor(openai_api_key, temp_dir, output_dir)
    else:
        raise ValueError("Approach must be 'simple' or 'structured'")
    
    return preprocessor.process(file_path, output_dir=output_dir)


def get_preprocessing_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of preprocessing results"""
    assessments = result['condition_assessments']
    
    summary = {
        'file_type': 'PDF' if result['is_pdf'] else 'Image',
        'total_pages': result['total_pages'],
        'processing_time': result['processing_time'],
        'pages_processed': result.get('pages_preprocessed', 0),
        'pages_skipped': result.get('pages_skipped', 0),
        'common_issues': {},
        'actions_applied': {},
        'average_success_rate': 0.0
    }
    
    if 'pages_preprocessed' not in result:
        processing_results = result.get('processing_results', [])
        summary['pages_processed'] = sum(1 for result in processing_results if result is not None)
        summary['pages_skipped'] = len(processing_results) - summary['pages_processed']
    
    success_rates = []
    
    for i, assessment_data in enumerate(assessments):
        # Ensure we are working with a Pydantic object for consistent access
        if isinstance(assessment_data, dict):
            assessment = DocumentConditionAssessment.model_validate(assessment_data)
        else:
            assessment = assessment_data

        for issue in assessment.primary_issues:
            issue_name = issue.condition.value
            summary['common_issues'][issue_name] = summary['common_issues'].get(issue_name, 0) + 1
        
        for action in assessment.processing_recommendations:
            action_name = action.value
            summary['actions_applied'][action_name] = summary['actions_applied'].get(action_name, 0) + 1
        
        success_rates.append(assessment.estimated_success_rate)
    
    if success_rates:
        summary['average_success_rate'] = sum(success_rates) / len(success_rates)
    
    return summary


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example usage of both approaches"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable required")
        return
    
    test_files = [
        "data/inputs/var3.pdf",
        # "data/inputs/poor_quality_scan.png"
    ]
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  Test file not found: {test_file}")
            continue
        
        print(f"\n=== Processing {test_file} ===")
        
        for approach in ["simple", "structured"]:
            print(f"\n--- Using {approach} approach ---")
            
            try:
                result = preprocess_document(
                    file_path=test_file,
                    openai_api_key=api_key,
                    output_dir="data/outputs/my_preprocessed_docs",
                    approach=approach
                )
                
                summary = get_preprocessing_summary(result)
                
                print(f"✅ Success! {summary['file_type']} with {summary['total_pages']} page(s)")
                print(f"📁 Output directory: {result['output_directory']}")
                print(f"Processing time: {summary['processing_time']:.2f}s")
                print(f"Pages processed: {summary['pages_processed']}")
                print(f"Pages skipped: {summary['pages_skipped']}")
                print(f"Average success rate: {summary['average_success_rate']:.2%}")
                
                print("📄 Preprocessed images:")
                for i, path in enumerate(result['processed_image_paths'], 1):
                    print(f"  Page {i}: {path}")
                
                if summary['common_issues']:
                    print("⚠️  Common issues detected:")
                    for issue, count in sorted(summary['common_issues'].items()):
                        print(f"  - {issue}: {count} page(s)")
                
                if summary['actions_applied']:
                    print("🔧 Actions applied:")
                    for action, count in sorted(summary['actions_applied'].items()):
                        print(f"  - {action}: {count} page(s)")
                
            except Exception as e:
                print(f"❌ Failed with {approach} approach: {e}")


if __name__ == "__main__":
    example_usage()
