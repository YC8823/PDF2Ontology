"""
Main document analysis orchestrator
"""

import os, sys
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# add directory to path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from .models import DocumentAnalysisResult, DocumentProcessingState, ProcessingStep, DocumentRegion
from .region_detector import RegionDetector
from ..utils.image_utils import ImageProcessor
from ..utils.validation import DocumentValidator
from ..utils.visualization import RegionVisualizer

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """Main orchestrator for document visual analysis pipeline"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        
        # Use structured output detector
        self.region_detector = RegionDetector(openai_api_key, model_name)
        self.visualizer = RegionVisualizer()
        
        self.current_state: Optional[DocumentProcessingState] = None
    
    def analyze_document(self, input_path: str, output_dir: str,
                        enhance_image: bool = True,
                        annotation_style: str = "detailed",
                        extract_tables: bool = False
                        ) -> Dict[str, Any]:
                       
                        #extract_content: bool = False
        """Analyze document using structured output for improved reliability"""
        
        document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_state = DocumentProcessingState(
            document_id=document_id,
            input_path=input_path,
            output_dir=output_dir,
            current_step="initialization"
        )
        
        try:
            # Step 1: Validate inputs
            self._update_step("validation", "running")
            validation_result = self._validate_inputs(input_path, output_dir)
            self._update_step("validation", "completed", validation_result)
            
            # Step 2: Preprocess image
            self._update_step("preprocessing", "running")
            processed_image_path = self._preprocess_image(input_path, enhance_image)
            self._update_step("preprocessing", "completed", {"processed_path": processed_image_path})
            
            # Step 3: Structured region detection
            self._update_step("structured_region_detection", "running")
            analysis_result = self.region_detector.detect_regions(
                image_path=processed_image_path,
                page_number=0,
                document_id=document_id
            )
            self._update_step("structured_region_detection", "completed", {
                "region_count": len(analysis_result.regions),
                "processing_metadata": analysis_result.processing_metadata
            })
            
            # Step 4: Table analysis (optional)
            table_results = {}
            if extract_tables:
                self._update_step("table_analysis", "running")
                table_results = self._analyze_tables_structured(processed_image_path, analysis_result.regions)
                self._update_step("table_analysis", "completed", {"tables_analyzed": len(table_results)})
            
            # # Step 5: Content extraction (optional)
            # content_results = {}
            # if extract_content:
            #     self._update_step("content_extraction", "running")
            #     content_results = self._extract_content_structured(processed_image_path, analysis_result.regions)
            #     self._update_step("content_extraction", "completed", content_results)
            
            # Step 6: Visualization
            self._update_step("visualization", "running")
            visualization_paths = self._create_visualizations(
                processed_image_path, analysis_result, output_dir, annotation_style
            )
            self._update_step("visualization", "completed", visualization_paths)
            
            # Step 7: Save results
            self._update_step("output_generation", "running")
            #output_paths = self._save_results(analysis_result, table_results, content_results, output_dir)
            output_paths = self._save_results(analysis_result, table_results, output_dir)
            self._update_step("output_generation", "completed", output_paths)
            
            # Compile final results
            final_result = {
                "analysis_result": analysis_result.model_dump(),
                "table_analysis": table_results,
                #"content_extraction": content_results,
                "processing_state": self.current_state.model_dump(),
                "output_files": {**output_paths, **visualization_paths},
                "metadata": {
                    "processing_time": self._calculate_total_processing_time(),
                    "input_file": input_path,
                    "output_directory": output_dir,
                    "structured_output_used": True,
                    "model_name": self.model_name
                }
            }
            
            logger.info(f"Enhanced analysis completed for {input_path}")
            return final_result
            
        except Exception as e:
            self._update_step(self.current_state.current_step, "failed", error_message=str(e))
            logger.error(f"Enhanced analysis failed: {str(e)}")
            raise
    
    def _analyze_tables_structured(self, image_path: str, regions: List[DocumentRegion]) -> Dict[str, Any]:
        """Analyze table structures using structured output"""
        
        table_results = {}
        table_regions = [r for r in regions if r.region_type.value == "table"]
        
        for i, table_region in enumerate(table_regions):
            try:
                table_structure = self.region_detector.analyze_table_structure(image_path, table_region)
                table_results[f"table_{i}"] = table_structure.dict()
                
            except Exception as e:
                logger.warning(f"Failed to analyze table {i}: {str(e)}")
                continue
        
        return table_results
    
    # def _extract_content_structured(self, image_path: str, regions: List[DocumentRegion]) -> Dict[str, Any]:
    #     """Extract content using structured output"""
        
    #     try:
    #         # Filter text regions
    #         text_regions = [r for r in regions if r.region_type.value in ["text", "title", "header", "footer"]]
            
    #         if not text_regions:
    #             return {"message": "No text regions found for content extraction"}
            
    #         content_result = self.region_detector.extract_content_structured(image_path, text_regions)
    #         return content_result.dict()
            
    #     except Exception as e:
    #         logger.warning(f"Content extraction failed: {str(e)}")
    #         return {"error": str(e)}

    def _update_step(self, step_name: str, status: str, result: Any = None, error_message: str = None):
        """Update processing step status"""
        
        # Find existing step or create new one
        step = next((s for s in self.current_state.steps if s.step_name == step_name), None)
        
        if step is None:
            step = ProcessingStep(step_name=step_name, status="pending")
            self.current_state.steps.append(step)
        
        # Update step
            if result is not None:
                step.result = result
            if error_message is not None:
                step.error_message = error_message
        
        # Update current step
        self.current_state.current_step = step_name
    
    def _validate_inputs(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """Validate input file and output directory"""
        
        # Validate input file
        file_metadata = DocumentValidator.validate_input_file(input_path)
        
        # Validate output directory
        validated_output_dir = DocumentValidator.validate_output_directory(output_dir)
        
        # Validate processing parameters
        params = {
            'openai_api_key': self.openai_api_key,
            'model_name': self.model_name
        }
        validated_params = DocumentValidator.validate_processing_params(params)
        
        return {
            "file_metadata": file_metadata,
            "output_directory": validated_output_dir,
            "processing_params": validated_params
        }
    
    def _preprocess_image(self, input_path: str, enhance: bool) -> str:
        """Preprocess image for optimal analysis"""
        
        if not enhance:
            return input_path
        
        # Resize if image too large, now size for A4; wrong annotation
        resized_path = ImageProcessor.resize_if_needed(input_path)
        
        # Enhance quality
        if enhance:
            enhanced_path = ImageProcessor.preprocess_image(resized_path)
            return enhanced_path
        
        return resized_path
    
    def _create_visualizations(self, image_path: str, analysis_result: DocumentAnalysisResult,
                             output_dir: str, style: str) -> Dict[str, str]:
        """Create visual annotations"""
        
        visualization_paths = {}
        
        # Main annotated image
        annotated_path = os.path.join(output_dir, "annotated_document.png")
        self.visualizer.annotate_regions(image_path, analysis_result.regions, annotated_path, style)
        visualization_paths["annotated_image"] = annotated_path
        
        # Region summary
        summary_path = os.path.join(output_dir, "region_summary.png")
        self.visualizer.create_region_summary(analysis_result.regions, summary_path)
        visualization_paths["summary_image"] = summary_path
        
        # Create different annotation styles
        for style_name in ["minimal", "confidence_heatmap"]:
            style_path = os.path.join(output_dir, f"annotated_{style_name}.png")
            self.visualizer.annotate_regions(image_path, analysis_result.regions, style_path, style_name)
            visualization_paths[f"annotated_{style_name}"] = style_path
        
        return visualization_paths
    
    def _save_results(self, analysis_result: DocumentAnalysisResult, table_results: Dict,output_dir: str) -> Dict[str, str]:
                     # content_results: Dict, 
        """Save all results"""
        output_paths = {}
        
        # Main results JSON
        json_path = os.path.join(output_dir, "structured_analysis_results.json")
        results_data = {
            "layout_analysis": analysis_result.model_dump(),
            "table_analysis": table_results,
            #"content_extraction": content_results,
            "metadata": {
                "analysis_method": "structured_output",
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
        output_paths["structured_results"] = json_path
        
        # Regions CSV
        csv_path = os.path.join(output_dir, "detected_regions.csv")
        self._save_regions_csv(analysis_result.regions, csv_path)
        output_paths["regions_csv"] = csv_path
        
        return output_paths
    
    # 论文专用 
    def _save_regions_csv(self, regions: List[DocumentRegion], csv_path: str):
        """Save regions to CSV"""
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['region_id', 'region_type', 'x', 'y', 'width', 'height', 
                         'confidence', 'content_description', 'reading_order']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for region in regions:
                writer.writerow({
                    'region_id': region.region_id,
                    'region_type': region.region_type.value,
                    'x': region.bbox.x,
                    'y': region.bbox.y,
                    'width': region.bbox.width,
                    'height': region.bbox.height,
                    'confidence': region.confidence,
                    'content_description': region.content_description,
                    'reading_order': region.reading_order
                })
    
    def _calculate_total_processing_time(self) -> float:
        """Calculate total processing time"""
        start_times = [s.start_time for s in self.current_state.steps if s.start_time]
        end_times = [s.end_time for s in self.current_state.steps if s.end_time]
        
        if start_times and end_times:
            total_start = min(start_times)
            total_end = max(end_times)
            return (total_end - total_start).total_seconds()
        
        return 0.0
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        
        if not self.current_state:
            return {"status": "not_started"}
        
        return {
            "document_id": self.current_state.document_id,
            "current_step": self.current_state.current_step,
            "steps": [
                {
                    "name": step.step_name,
                    "status": step.status,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "error": step.error_message
                }
                for step in self.current_state.steps
            ],
            "progress": len([s for s in self.current_state.steps if s.status == "completed"]) / max(len(self.current_state.steps), 1)
        }