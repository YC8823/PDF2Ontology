"""
Streamlined document analysis orchestrator
Integrates visual analysis and table processing
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..pydantic_models.region_models import DocumentLayout, DocumentRegion
from ..pydantic_models.table_models import TableData
from ..pydantic_models.enums import RegionType
from .region_detector import RegionDetector
from ..table_processors.table_transformer import TableTransformerProcessor
from ..utils.image_utils import convert_pdf_to_images
from ..utils.validation import DocumentValidator
from ..utils.visualization import RegionVisualizer

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """Streamlined document analysis orchestrator"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        
        # Initialize core components
        self.region_detector = RegionDetector(openai_api_key, model_name)
        self.table_processor = TableTransformerProcessor()
        self.visualizer = RegionVisualizer()
    
    def analyze_document(self, 
                        input_path: str, 
                        output_dir: str,
                        extract_tables: bool = True,
                        create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Analyze document with visual layout detection and table processing
        
        Args:
            input_path: Path to input file (image or PDF)
            output_dir: Output directory path
            extract_tables: Whether to perform detailed table structure analysis
            create_visualizations: Whether to create annotated visualizations
            
        Returns:
            Analysis results dictionary
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Validate inputs and prepare output directory
            self._prepare_analysis(input_path, output_dir)
            
            # Step 2: Convert input to images if needed
            image_paths = self._prepare_images(input_path)
            
            # Step 3: Process each page
            all_results = []
            for page_num, image_path in enumerate(image_paths, 1):
                logger.info(f"Processing page {page_num}/{len(image_paths)}")
                
                page_result = self._process_single_page(
                    image_path, page_num, extract_tables
                )
                all_results.append(page_result)
            
            # Step 4: Compile results
            final_result = self._compile_results(
                all_results, input_path, output_dir, start_time
            )
            
            # Step 5: Create visualizations if requested
            if create_visualizations:
                visualization_paths = self._create_visualizations(
                    image_paths, all_results, output_dir
                )
                final_result["visualizations"] = visualization_paths
            
            # Step 6: Save results
            output_paths = self._save_results(final_result, output_dir)
            final_result["output_files"] = output_paths
            
            logger.info(f"Document analysis completed for {input_path}")
            return final_result
            
        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            raise
    
    def _prepare_analysis(self, input_path: str, output_dir: str) -> None:
        """Validate inputs and prepare output directory"""
        # Validate input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analysis prepared - Input: {input_path}, Output: {output_dir}")
    
    def _prepare_images(self, input_path: str) -> List[str]:
        """Convert input to list of image paths"""
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext == '.pdf':
            logger.info("Converting PDF to images...")
            pdf_images = convert_pdf_to_images(input_path, dpi=300)
            
            if not pdf_images:
                raise ValueError("Failed to convert PDF to images")
            
            # Save images temporarily
            import tempfile
            temp_dir = tempfile.mkdtemp()
            image_paths = []
            
            for i, img in enumerate(pdf_images):
                temp_path = os.path.join(temp_dir, f"page_{i+1}.png")
                img.save(temp_path, 'PNG')
                image_paths.append(temp_path)
            
            logger.info(f"Converted PDF to {len(image_paths)} images")
            return image_paths
            
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return [input_path]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _process_single_page(self, 
                           image_path: str, 
                           page_number: int,
                           extract_tables: bool) -> Dict[str, Any]:
        """Process a single page for layout and table analysis"""
        
        # Step 1: Visual layout analysis
        logger.info(f"  Analyzing layout for page {page_number}...")
        layout = self.region_detector.analyze_document(image_path)
        
        # Update page numbers in regions
        for region in layout.regions:
            region.page_number = page_number
        
        page_result = {
            "page_number": page_number,
            "layout": layout,
            "table_analysis": {}
        }
        
        # Step 2: Table structure analysis if requested
        if extract_tables:
            table_regions = [r for r in layout.regions if r.type == RegionType.TABLE]
            
            if table_regions:
                logger.info(f"  Found {len(table_regions)} table regions on page {page_number}")
                page_result["table_analysis"] = self._analyze_page_tables(
                    image_path, table_regions, page_number
                )
            else:
                logger.info(f"  No table regions found on page {page_number}")
        
        return page_result
    
    def _analyze_page_tables(self, 
                           image_path: str, 
                           table_regions: List[DocumentRegion],
                           page_number: int) -> Dict[str, Any]:
        """Analyze table structures on a single page"""
        
        table_results = {}
        
        for i, table_region in enumerate(table_regions):
            table_id = f"page_{page_number}_table_{i+1}"
            logger.info(f"    Processing {table_id}...")
            
            try:
                # Convert region bbox to dict format for table processor
                bbox_dict = {
                    'x': table_region.bbox.x,
                    'y': table_region.bbox.y,
                    'width': table_region.bbox.width,
                    'height': table_region.bbox.height
                }
                
                # Use Table Transformer for detailed structure analysis
                table_structure = self.table_processor.detect_table_structure(
                    image_path=image_path,
                    bbox=bbox_dict,
                    page_number=page_number
                )
                
                # Get detection summary
                summary = self.table_processor.get_detection_summary(table_structure)
                
                table_results[table_id] = {
                    "visual_detection": {
                        "region_id": table_region.id,
                        "confidence": table_region.confidence,
                        "content_description": table_region.content_description,
                        "bbox": bbox_dict
                    },
                    "structure_analysis": table_structure.model_dump(),
                    "detection_summary": summary,
                    "combined_confidence": (
                        table_region.confidence + table_structure.structure_confidence
                    ) / 2
                }
                
                logger.info(f"      Structure: {summary['dimensions']}, "
                          f"confidence: {table_results[table_id]['combined_confidence']:.2f}")
                
            except Exception as e:
                logger.warning(f"    Failed to analyze {table_id}: {str(e)}")
                table_results[table_id] = {
                    "error": str(e),
                    "visual_detection": {
                        "region_id": table_region.id,
                        "confidence": table_region.confidence,
                        "bbox": bbox_dict
                    }
                }
        
        return table_results
    
    def _compile_results(self, 
                        page_results: List[Dict[str, Any]], 
                        input_path: str,
                        output_dir: str,
                        start_time: datetime) -> Dict[str, Any]:
        """Compile all page results into final analysis result"""
        
        # Aggregate statistics
        total_regions = sum(len(page["layout"].regions) for page in page_results)
        total_tables = sum(len(page["table_analysis"]) for page in page_results)
        
        # Calculate successful table analyses
        successful_tables = 0
        for page in page_results:
            for table_data in page["table_analysis"].values():
                if "error" not in table_data:
                    successful_tables += 1
        
        # Compile final result
        final_result = {
            "document_info": {
                "input_path": input_path,
                "output_directory": output_dir,
                "total_pages": len(page_results),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            },
            "analysis_summary": {
                "total_regions_detected": total_regions,
                "total_table_regions": total_tables,
                "successful_table_analyses": successful_tables,
                "analysis_method": "visual_analyzer + table_transformer",
                "model_used": self.model_name
            },
            "page_results": page_results,
            "metadata": {
                "analyzer_version": "2.0",
                "components_used": ["RegionDetector", "TableTransformerProcessor"],
                "success_rate": successful_tables / max(total_tables, 1)
            }
        }
        
        return final_result
    
    def _create_visualizations(self, 
                             image_paths: List[str],
                             page_results: List[Dict[str, Any]],
                             output_dir: str) -> Dict[str, List[str]]:
        """Create visualizations for all pages"""
        
        visualization_paths = {
            "annotated_pages": [],
            "table_highlights": [],
            "summary_charts": []
        }
        
        for i, (image_path, page_result) in enumerate(zip(image_paths, page_results)):
            page_num = i + 1
            
            # Create annotated page
            annotated_path = os.path.join(output_dir, f"page_{page_num}_annotated.png")
            self.visualizer.annotate_regions(
                image_path, page_result["layout"].regions, annotated_path, "detailed"
            )
            visualization_paths["annotated_pages"].append(annotated_path)
            
            # Create table-specific visualization if tables exist
            if page_result["table_analysis"]:
                table_highlight_path = os.path.join(output_dir, f"page_{page_num}_tables.png")
                table_regions = [r for r in page_result["layout"].regions if r.type == RegionType.TABLE]
                self.visualizer.annotate_regions(
                    image_path, table_regions, table_highlight_path, "minimal"
                )
                visualization_paths["table_highlights"].append(table_highlight_path)
        
        # Create summary visualization
        summary_path = os.path.join(output_dir, "analysis_summary.png")
        all_regions = []
        for page_result in page_results:
            all_regions.extend(page_result["layout"].regions)
        
        self.visualizer.create_region_summary(all_regions, summary_path)
        visualization_paths["summary_charts"].append(summary_path)
        
        return visualization_paths
    
    def _save_results(self, final_result: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Save analysis results to files"""
        
        output_paths = {}
        
        # Save main results as JSON
        results_path = os.path.join(output_dir, "document_analysis_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)
        output_paths["main_results"] = results_path
        
        # Save regions summary as CSV
        csv_path = os.path.join(output_dir, "detected_regions.csv")
        self._save_regions_csv(final_result["page_results"], csv_path)
        output_paths["regions_csv"] = csv_path
        
        # Save table analysis summary
        if any(page["table_analysis"] for page in final_result["page_results"]):
            table_summary_path = os.path.join(output_dir, "table_analysis_summary.json")
            table_summary = self._create_table_summary(final_result["page_results"])
            with open(table_summary_path, 'w', encoding='utf-8') as f:
                json.dump(table_summary, f, indent=2, ensure_ascii=False, default=str)
            output_paths["table_summary"] = table_summary_path
        
        return output_paths
    
    def _save_regions_csv(self, page_results: List[Dict[str, Any]], csv_path: str) -> None:
        """Save all detected regions to CSV"""
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'page_number', 'region_id', 'region_type', 'x', 'y', 'width', 'height',
                'confidence', 'content_description', 'reading_order'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for page_result in page_results:
                page_num = page_result["page_number"]
                for region in page_result["layout"].regions:
                    writer.writerow({
                        'page_number': page_num,
                        'region_id': region.id,
                        'region_type': region.type.value,
                        'x': region.bbox.x,
                        'y': region.bbox.y,
                        'width': region.bbox.width,
                        'height': region.bbox.height,
                        'confidence': region.confidence,
                        'content_description': region.content_description,
                        'reading_order': getattr(region, 'reading_order', None)
                    })
    
    def _create_table_summary(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of table analysis results"""
        
        summary = {
            "total_tables_detected": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "table_details": [],
            "statistics": {
                "avg_confidence": 0.0,
                "table_sizes": [],
                "pages_with_tables": 0
            }
        }
        
        confidence_scores = []
        pages_with_tables = set()
        
        for page_result in page_results:
            page_num = page_result["page_number"]
            
            for table_id, table_data in page_result["table_analysis"].items():
                summary["total_tables_detected"] += 1
                
                if "error" in table_data:
                    summary["failed_analyses"] += 1
                    summary["table_details"].append({
                        "table_id": table_id,
                        "page": page_num,
                        "status": "failed",
                        "error": table_data["error"]
                    })
                else:
                    summary["successful_analyses"] += 1
                    pages_with_tables.add(page_num)
                    
                    conf = table_data["combined_confidence"]
                    confidence_scores.append(conf)
                    
                    struct_summary = table_data["detection_summary"]
                    summary["table_details"].append({
                        "table_id": table_id,
                        "page": page_num,
                        "status": "success",
                        "dimensions": struct_summary["dimensions"],
                        "confidence": conf,
                        "has_headers": struct_summary["header_cells"] > 0,
                        "has_merged_cells": struct_summary["merged_cells"] > 0
                    })
                    
                    # Extract table size for statistics
                    dims = struct_summary["dimensions"].split("x")
                    if len(dims) == 2:
                        rows, cols = int(dims[0]), int(dims[1])
                        summary["statistics"]["table_sizes"].append(rows * cols)
        
        # Calculate statistics
        if confidence_scores:
            summary["statistics"]["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        summary["statistics"]["pages_with_tables"] = len(pages_with_tables)
        
        return summary

# ==================== Usage Example ====================
def example_usage():
    """Example usage of the refactored DocumentAnalyzer"""

    from dotenv import load_dotenv
    load_dotenv()
    
    # Configuration
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable required")
        print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False

    # Initialize analyzer
    analyzer = DocumentAnalyzer(api_key)  # Replace with actual API key
    
    # Analyze a PDF document
    input_path = "data/inputs/sample02.pdf"
    output_dir = "data/outputs/analysis_results"
    
    try:
        # Run analysis
        results = analyzer.analyze_document(
            input_path=input_path,
            output_dir=output_dir,
            extract_tables=True,
            create_visualizations=True
        )
        
        # Print summary
        print("=== Document Analysis Complete ===")
        print(f"Total pages: {results['document_info']['total_pages']}")
        print(f"Processing time: {results['document_info']['processing_time']:.2f}s")
        print(f"Total regions: {results['analysis_summary']['total_regions_detected']}")
        print(f"Table regions: {results['analysis_summary']['total_table_regions']}")
        print(f"Successful table analyses: {results['analysis_summary']['successful_table_analyses']}")
        print(f"Success rate: {results['metadata']['success_rate']:.2%}")
        
        # Show table details
        if results['analysis_summary']['total_table_regions'] > 0:
            print("\n=== Table Analysis Details ===")
            for page_result in results['page_results']:
                page_num = page_result['page_number']
                tables = page_result['table_analysis']
                
                if tables:
                    print(f"\nPage {page_num}:")
                    for table_id, table_data in tables.items():
                        if "error" not in table_data:
                            summary = table_data['detection_summary']
                            conf = table_data['combined_confidence']
                            print(f"  {table_id}: {summary['dimensions']}, confidence: {conf:.2f}")
                        else:
                            print(f"  {table_id}: Failed - {table_data['error']}")
        
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    example_usage()