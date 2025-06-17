import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ..pydantic_models.table_models import TableStructure, TableCell, TableHeader
from ..pydantic_models.region_models import BoundingBox
from ..pydantic_models.enums import CellType, ExtractionMethod
import uuid

class TableTransformerProcessor:
    """Table Transformer model processor with Pydantic model integration"""
    
    def __init__(self, model_name: str = "microsoft/table-transformer-structure-recognition"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Category mapping
        self.id2label = {
            0: "table",
            1: "table column",
            2: "table row", 
            3: "table column header",
            4: "table projected row header",
            5: "table spanning cell"
        }
    
    def detect_table_structure(self, image_path: str, 
                              bbox: Dict[str, float],
                              page_number: int = 1) -> TableStructure:
        """
        Detect table structure and return structured TableStructure object
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box of the table region (relative coordinates)
            page_number: Page number where table is located
            
        Returns:
            TableStructure: Structured table information using Pydantic model
        """
        # Load and crop image to table region
        image = Image.open(image_path).convert("RGB")
        table_image = self._crop_table_region(image, bbox)
        
        # Preprocess image
        inputs = self.processor(images=table_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Model inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([table_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]
        
        # Parse detection results into structured format
        detection_data = self._parse_detection_results(results, table_image.size)
        
        # Convert to TableStructure object
        table_structure = self._create_table_structure(
            detection_data, bbox, page_number, table_image.size
        )
        
        return table_structure
    
    def _crop_table_region(self, image: Image.Image, bbox: Dict[str, float]) -> Image.Image:
        """Crop table region from full image"""
        img_width, img_height = image.size
        
        # Convert relative coordinates to absolute coordinates
        x = int(bbox['x'] * img_width)
        y = int(bbox['y'] * img_height)
        width = int(bbox['width'] * img_width)
        height = int(bbox['height'] * img_height)
        
        # Crop image
        cropped = image.crop((x, y, x + width, y + height))
        return cropped
    
    def _parse_detection_results(self, results: Dict[str, Any], 
                               image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Parse detection results from the model"""
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        # Group by category
        detected_objects = {
            "rows": [],
            "columns": [],
            "cells": [],
            "headers": [],
            "spanning_cells": []
        }
        
        for box, score, label in zip(boxes, scores, labels):
            # Convert box coordinates to relative coordinates
            rel_box = self._convert_to_relative_bbox(box, image_size)
            
            obj_info = {
                "bbox": rel_box,
                "confidence": float(score),
                "label": self.id2label.get(int(label), "unknown"),
                "label_id": int(label)
            }
            
            if label == 2:  # table row
                detected_objects["rows"].append(obj_info)
            elif label == 1:  # table column
                detected_objects["columns"].append(obj_info)
            elif label == 3:  # table column header
                detected_objects["headers"].append(obj_info)
            elif label == 4:  # table projected row header
                detected_objects["headers"].append(obj_info)
            elif label == 5:  # table spanning cell
                detected_objects["spanning_cells"].append(obj_info)
        
        # Compute grid structure
        grid_structure = self._compute_grid_structure(detected_objects)
        
        return {
            "detected_objects": detected_objects,
            "grid_structure": grid_structure,
            "image_size": image_size
        }
    
    def _convert_to_relative_bbox(self, box: np.ndarray, 
                                 image_size: Tuple[int, int]) -> BoundingBox:
        """Convert absolute bbox coordinates to relative BoundingBox object"""
        width, height = image_size
        x1, y1, x2, y2 = box
        
        return BoundingBox(
            x=float(x1 / width),
            y=float(y1 / height),
            width=float((x2 - x1) / width),
            height=float((y2 - y1) / height)
        )
    
    def _compute_grid_structure(self, detected_objects: Dict[str, List]) -> Dict[str, Any]:
        """Compute table grid structure"""
        rows = detected_objects["rows"]
        columns = detected_objects["columns"]
        
        # Sort by position
        rows_sorted = sorted(rows, key=lambda x: x["bbox"].y)  # Sort by y coordinate
        cols_sorted = sorted(columns, key=lambda x: x["bbox"].x)  # Sort by x coordinate
        
        return {
            "num_rows": len(rows_sorted),
            "num_cols": len(cols_sorted),
            "row_positions": [r["bbox"] for r in rows_sorted],
            "col_positions": [c["bbox"] for c in cols_sorted],
            "has_spanning_cells": len(detected_objects["spanning_cells"]) > 0
        }
    
    def _create_table_structure(self, detection_data: Dict[str, Any], 
                               original_bbox: Dict[str, float],
                               page_number: int,
                               cropped_size: Tuple[int, int]) -> TableStructure:
        """Create TableStructure object from detection data"""
        
        grid = detection_data["grid_structure"]
        detected_objects = detection_data["detected_objects"]
        
        # Create table cells based on grid structure
        cells = self._generate_table_cells(grid, detected_objects)
        
        # Extract headers
        headers = self._extract_headers(detected_objects["headers"], grid)
        
        # Create multi-level headers if needed
        multi_level_headers = self._create_multi_level_headers(detected_objects["headers"])
        
        # Convert original bbox to BoundingBox object
        table_bbox = BoundingBox(
            x=original_bbox['x'],
            y=original_bbox['y'],
            width=original_bbox['width'],
            height=original_bbox['height']
        )
        
        # Calculate structure confidence
        structure_confidence = self._calculate_structure_confidence(detected_objects)
        
        return TableStructure(
            rows=max(grid["num_rows"], 1),
            cols=max(grid["num_cols"], 1),
            cells=cells,
            headers=headers,
            multi_level_headers=multi_level_headers,
            page_number=page_number,
            bbox=table_bbox,
            extraction_method=ExtractionMethod.TABLE_TRANSFORMER,
            structure_confidence=structure_confidence,
            has_merged_cells=grid["has_spanning_cells"],
            table_type="detected_structure",
            metadata={
                "detection_data": detection_data,
                "cropped_image_size": cropped_size,
                "model_name": "microsoft/table-transformer-structure-recognition"
            }
        )
    
    def _generate_table_cells(self, grid: Dict[str, Any], 
                             detected_objects: Dict[str, List]) -> List[TableCell]:
        """Generate table cells based on grid structure"""
        cells = []
        
        # Create basic grid cells
        for row in range(grid["num_rows"]):
            for col in range(grid["num_cols"]):
                # Determine cell type
                cell_type = self._determine_cell_type(row, col, detected_objects)
                
                # Calculate cell bbox (approximation based on grid)
                cell_bbox = self._calculate_cell_bbox(row, col, grid)
                
                cell = TableCell(
                    row=row,
                    col=col,
                    content="",  # Content will be filled by content extractor
                    cell_type=cell_type,
                    confidence=0.8,  # Default confidence from structure detection
                    bbox=cell_bbox,
                    metadata={
                        "generated_from": "grid_structure",
                        "grid_position": {"row": row, "col": col}
                    }
                )
                cells.append(cell)
        
        # Handle spanning cells
        self._process_spanning_cells(cells, detected_objects["spanning_cells"])
        
        return cells
    
    def _determine_cell_type(self, row: int, col: int, 
                           detected_objects: Dict[str, List]) -> CellType:
        """Determine cell type based on position and detected headers"""
        # Check if this position corresponds to a header
        for header in detected_objects["headers"]:
            header_bbox = header["bbox"]
            # Simple heuristic: if it's in the top rows, it's likely a header
            if row == 0 or (row == 1 and len(detected_objects["headers"]) > 0):
                return CellType.HEADER
        
        return CellType.DATA
    
    def _calculate_cell_bbox(self, row: int, col: int, 
                           grid: Dict[str, Any]) -> Optional[BoundingBox]:
        """Calculate approximate cell bounding box"""
        if not grid["row_positions"] or not grid["col_positions"]:
            return None
            
        if row >= len(grid["row_positions"]) or col >= len(grid["col_positions"]):
            return None
        
        # Get row and column boundaries
        if row < len(grid["row_positions"]):
            row_bbox = grid["row_positions"][row]
            row_y = row_bbox.y
            row_height = row_bbox.height
        else:
            return None
            
        if col < len(grid["col_positions"]):
            col_bbox = grid["col_positions"][col]
            col_x = col_bbox.x
            col_width = col_bbox.width
        else:
            return None
        
        return BoundingBox(
            x=col_x,
            y=row_y,
            width=col_width,
            height=row_height
        )
    
    def _process_spanning_cells(self, cells: List[TableCell], 
                              spanning_cells: List[Dict[str, Any]]) -> None:
        """Process spanning cells and update cell spans"""
        for span_info in spanning_cells:
            span_bbox = span_info["bbox"]
            
            # Find cells that this spanning cell covers
            covered_cells = []
            for cell in cells:
                if cell.bbox and self._bbox_overlap(span_bbox, cell.bbox):
                    covered_cells.append(cell)
            
            if covered_cells:
                # Update the first cell to be the spanning cell
                main_cell = covered_cells[0]
                
                # Calculate span dimensions
                min_row = min(cell.row for cell in covered_cells)
                max_row = max(cell.row for cell in covered_cells)
                min_col = min(cell.col for cell in covered_cells)
                max_col = max(cell.col for cell in covered_cells)
                
                main_cell.rowspan = max_row - min_row + 1
                main_cell.colspan = max_col - min_col + 1
                main_cell.cell_type = CellType.MERGED
                main_cell.confidence = span_info["confidence"]
                
                # Mark other cells as part of the span
                for cell in covered_cells[1:]:
                    cell.cell_type = CellType.EMPTY
                    cell.metadata["part_of_span"] = main_cell.id
    
    def _bbox_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
        """Check if two bounding boxes overlap"""
        return not (bbox1.right <= bbox2.x or bbox2.right <= bbox1.x or
                   bbox1.bottom <= bbox2.y or bbox2.bottom <= bbox1.y)
    
    def _extract_headers(self, header_objects: List[Dict[str, Any]], 
                        grid: Dict[str, Any]) -> List[str]:
        """Extract header text (placeholder - actual text extraction done by content extractor)"""
        headers = []
        
        # Sort headers by column position
        sorted_headers = sorted(header_objects, key=lambda x: x["bbox"].x)
        
        for i, header in enumerate(sorted_headers):
            headers.append(f"Column_{i+1}")  # Placeholder names
        
        # Ensure we have enough headers for all columns
        while len(headers) < grid["num_cols"]:
            headers.append(f"Column_{len(headers)+1}")
        
        return headers[:grid["num_cols"]]
    
    def _create_multi_level_headers(self, header_objects: List[Dict[str, Any]]) -> List[TableHeader]:
        """Create multi-level headers if detected"""
        multi_headers = []
        
        # Group headers by vertical position (y-coordinate)
        header_levels = {}
        for header in header_objects:
            y_pos = header["bbox"].y
            level = round(y_pos * 10)  # Discretize y-position
            
            if level not in header_levels:
                header_levels[level] = []
            header_levels[level].append(header)
        
        # Create TableHeader objects for each level
        for level_idx, (y_level, level_headers) in enumerate(sorted(header_levels.items())):
            for header in level_headers:
                # Estimate which columns this header spans
                header_bbox = header["bbox"]
                estimated_cols = self._estimate_column_span(header_bbox)
                
                table_header = TableHeader(
                    level=level_idx,
                    columns=estimated_cols,
                    text=f"Header_L{level_idx}",  # Placeholder
                    alignment="center"
                )
                multi_headers.append(table_header)
        
        return multi_headers
    
    def _estimate_column_span(self, header_bbox: BoundingBox) -> List[int]:
        """Estimate which columns a header spans"""
        # This is a simplified estimation
        # In practice, you'd compare with column positions
        start_col = int(header_bbox.x * 10)  # Simple heuristic
        end_col = int((header_bbox.x + header_bbox.width) * 10)
        
        return list(range(start_col, end_col + 1))
    
    def _calculate_structure_confidence(self, detected_objects: Dict[str, List]) -> float:
        """Calculate overall structure confidence"""
        all_confidences = []
        
        for obj_list in detected_objects.values():
            for obj in obj_list:
                all_confidences.append(obj["confidence"])
        
        if not all_confidences:
            return 0.5
        
        return float(np.mean(all_confidences))
    
    def get_detection_summary(self, table_structure: TableStructure) -> Dict[str, Any]:
        """Get a summary of the detection results"""
        return {
            "table_id": table_structure.id,
            "dimensions": f"{table_structure.rows}x{table_structure.cols}",
            "total_cells": len(table_structure.cells),
            "header_cells": len([c for c in table_structure.cells if c.cell_type == CellType.HEADER]),
            "merged_cells": len([c for c in table_structure.cells if c.cell_type == CellType.MERGED]),
            "structure_confidence": table_structure.structure_confidence,
            "has_multi_level_headers": len(table_structure.multi_level_headers) > 0,
            "extraction_method": table_structure.extraction_method,
            "page_number": table_structure.page_number
        }

# ==================== Usage Example ====================
def example_usage():
    """Example of how to use the improved TableTransformerProcessor"""
    
    # Initialize processor
    processor = TableTransformerProcessor()
    
    # Example table bbox (relative coordinates)
    table_bbox = {
        'x': 0.1,
        'y': 0.2,
        'width': 0.8,
        'height': 0.6
    }
    
    # Detect table structure
    table_structure = processor.detect_table_structure(
        image_path="path/to/document.png",
        bbox=table_bbox,
        page_number=1
    )
    
    # Get detection summary
    summary = processor.get_detection_summary(table_structure)
    print(f"Detected table: {summary}")
    
    # Access structured data
    print(f"Table has {table_structure.rows} rows and {table_structure.cols} columns")
    print(f"Headers: {table_structure.headers}")
    print(f"Confidence: {table_structure.structure_confidence:.2f}")
    
    # Access individual cells
    for cell in table_structure.cells:
        if cell.cell_type == CellType.HEADER:
            print(f"Header cell at ({cell.row}, {cell.col})")
        elif cell.cell_type == CellType.MERGED:
            print(f"Merged cell at ({cell.row}, {cell.col}) spans {cell.rowspan}x{cell.colspan}")

def example_usage_pdf():
    """Example of how to use the improved TableTransformerProcessor with PDF input"""
    
    # Import the PDF conversion utility
    from ..utils.image_utils import convert_pdf_to_images
    import tempfile
    import os
    
    # Initialize processor
    processor = TableTransformerProcessor()
    
    # PDF file path
    pdf_path = "data/inputs/sample02.pdf"  # Replace with actual PDF path
    
    # Convert PDF to images
    print("Converting PDF to images...")
    try:
        pdf_images = convert_pdf_to_images(pdf_path, dpi=300)
        
        if not pdf_images:
            print("Failed to convert PDF to images")
            return
        
        print(f"Successfully converted PDF to {len(pdf_images)} page images")
        
        # Process each page
        all_table_structures = []
        
        for page_num, page_image in enumerate(pdf_images, 1):
            print(f"\n--- Processing Page {page_num} ---")
            
            # Save the PIL image temporarily for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                page_image.save(temp_file.name, 'PNG')
                temp_image_path = temp_file.name
            
            try:
                # Example table bbox (relative coordinates)
                # You might want to detect table regions first using your visual analyzer
                table_bboxes = [
                    {
                        'x': 0.1,   # Left margin
                        'y': 0.05,   # Top margin
                        'width': 0.8,   # 80% of page width
                        'height': 0.6   # 30% of page height
                    },
                    # You can add more table regions if needed
                    # {
                    #     'x': 0.1,
                    #     'y': 0.6,
                    #     'width': 0.8,
                    #     'height': 0.3
                    # }
                ]
                
                page_tables = []
                
                for table_idx, table_bbox in enumerate(table_bboxes):
                    print(f"  Detecting table {table_idx + 1} on page {page_num}...")
                    
                    try:
                        # Detect table structure
                        table_structure = processor.detect_table_structure(
                            image_path=temp_image_path,
                            bbox=table_bbox,
                            page_number=page_num
                        )
                        
                        # Get detection summary
                        summary = processor.get_detection_summary(table_structure)
                        print(f"    Detected table: {summary}")
                        
                        # Only keep tables with reasonable structure
                        if table_structure.rows > 0 and table_structure.cols > 0:
                            page_tables.append(table_structure)
                            all_table_structures.append(table_structure)
                            
                            # Display detailed information
                            print(f"    Table has {table_structure.rows} rows and {table_structure.cols} columns")
                            print(f"    Headers: {table_structure.headers}")
                            print(f"    Structure confidence: {table_structure.structure_confidence:.2f}")
                            print(f"    Has merged cells: {table_structure.has_merged_cells}")
                            
                            # Show cell information
                            header_cells = [c for c in table_structure.cells if c.cell_type == CellType.HEADER]
                            merged_cells = [c for c in table_structure.cells if c.cell_type == CellType.MERGED]
                            
                            if header_cells:
                                print(f"    Found {len(header_cells)} header cells")
                            if merged_cells:
                                print(f"    Found {len(merged_cells)} merged cells:")
                                for cell in merged_cells:
                                    print(f"      - Cell at ({cell.row}, {cell.col}) spans {cell.rowspan}x{cell.colspan}")
                        else:
                            print(f"    Skipped table {table_idx + 1}: insufficient structure detected")
                            
                    except Exception as e:
                        print(f"    Error processing table {table_idx + 1}: {e}")
                
                print(f"  Found {len(page_tables)} valid tables on page {page_num}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
        
        # Summary of all detected tables
        print(f"\n=== Overall Summary ===")
        print(f"Total pages processed: {len(pdf_images)}")
        print(f"Total tables detected: {len(all_table_structures)}")
        
        if all_table_structures:
            # Statistics
            total_cells = sum(len(table.cells) for table in all_table_structures)
            avg_confidence = sum(table.structure_confidence for table in all_table_structures) / len(all_table_structures)
            tables_with_headers = sum(1 for table in all_table_structures if len(table.headers) > 0)
            tables_with_merged_cells = sum(1 for table in all_table_structures if table.has_merged_cells)
            
            print(f"Total cells detected: {total_cells}")
            print(f"Average structure confidence: {avg_confidence:.2f}")
            print(f"Tables with headers: {tables_with_headers}")
            print(f"Tables with merged cells: {tables_with_merged_cells}")
            
            # Show table distribution by page
            page_counts = {}
            for table in all_table_structures:
                page = table.page_number
                page_counts[page] = page_counts.get(page, 0) + 1
            
            print("\nTables per page:")
            for page, count in sorted(page_counts.items()):
                print(f"  Page {page}: {count} table(s)")
                
            # Show largest table
            largest_table = max(all_table_structures, key=lambda t: t.rows * t.cols)
            print(f"\nLargest table: {largest_table.rows}x{largest_table.cols} on page {largest_table.page_number}")
            
        else:
            print("No valid tables were detected in the PDF")
            
    except Exception as e:
        print(f"Error processing PDF: {e}")

# def example_usage_with_visual_analyzer():
#     """Advanced example using visual analyzer to detect table regions first"""
    
#     from ..utils.image_utils import convert_pdf_to_images
#     from ..core.document_analyzer import DocumentAnalyzer
#     from ..core.region_detector import RegionDetector
#     from ..pydantic_models.enums import RegionType
#     import tempfile
#     import os

#     from dotenv import load_dotenv
#     load_dotenv()
    
#     # Configuration
#     api_key = os.getenv('OPENAI_API_KEY')
#     if not api_key:
#         print("❌ Error: OPENAI_API_KEY environment variable required")
#         print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
#         return False
    
#     # Initialize processors
#     table_processor = TableTransformerProcessor()
#     visual_analyzer = DocumentAnalyzer(api_key)  
    
#     pdf_path = "data/inputs/sample_Datenblatt.pdf"  
    
#     print("Converting PDF to images...")
#     pdf_images = convert_pdf_to_images(pdf_path, dpi=300)
    
#     # Initialize converter and convert PDF
#     converter = PDFConverter()
#     image_paths = converter.convert_pdf_to_images(pdf_path, dpi=300)
    
#     if not image_paths:
#         print("❌ Failed to convert PDF to images")
#         return
    
#     print(f"Successfully converted {len(image_paths)} pages.")
    
#     # Initialize region detector
#     region_detector = RegionDetector(api_key)
    
#     # Process each page
#     for page_num, image_path in enumerate(image_paths, 1):
#         print(f"\n--- Processing Page {page_num} ---")
#         print("  Detecting document regions...")
        
#         try:
#             # Analyze document layout - this returns a DocumentLayout object
#             layout: DocumentLayout = region_detector.analyze_document(
#                 image_path=image_path, 
#                 page_number=page_num
#             )
            
#             print(f"  ✅ Detected {len(layout.regions)} regions")
            
#             # Extract table regions using the correct attribute name
#             table_regions = [region for region in layout.regions if region.type == RegionType.TABLE]
            
#             if not table_regions:
#                 print("  ℹ️  No table regions detected on this page")
#                 continue
            
#             print(f"  📊 Found {len(table_regions)} table region(s)")
            
#             # Process each table region
#             for i, table_region in enumerate(table_regions):
#                 print(f"\n  Table {i+1}:")
#                 print(f"    Region ID: {table_region.id}")
#                 print(f"    Confidence: {table_region.confidence:.2f}")
#                 print(f"    Description: {table_region.content_description}")
#                 print(f"    Bbox: ({table_region.bbox.x:.3f}, {table_region.bbox.y:.3f}, "
#                       f"{table_region.bbox.width:.3f}, {table_region.bbox.height:.3f})")
                
#                 # Analyze table structure if needed
#                 try:
#                     table_structure = region_detector.analyze_table_structure(image_path, table_region)
#                     print(f"    Structure: {table_structure.rows}x{table_structure.cols} table")
#                     print(f"    Has headers: {table_structure.headers}")
#                     print(f"    Structure confidence: {table_structure.structure_confidence:.2f}")
                    
#                     # Display some cell content if available
#                     if table_structure.cells:
#                         print(f"    Sample cells:")
#                         for j, cell in enumerate(table_structure.cells[:3]):  # Show first 3 cells
#                             print(f"      Cell {j+1}: '{cell.content}' at ({cell.row}, {cell.col})")
                        
#                         if len(table_structure.cells) > 3:
#                             print(f"      ... and {len(table_structure.cells) - 3} more cells")
                    
#                 except Exception as e:
#                     print(f"    ❌ Table structure analysis failed: {e}")
            
#             # Show summary of all regions detected
#             print(f"\n  📋 Region Summary for Page {page_num}:")
#             region_counts = {}
#             for region in layout.regions:
#                 region_type = region.type.value
#                 region_counts[region_type] = region_counts.get(region_type, 0) + 1
            
#             for region_type, count in sorted(region_counts.items()):
#                 print(f"    {region_type}: {count}")
                
#         except Exception as e:
#             print(f"  ❌ Failed to process page {page_num}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     print("\n🎉 Processing completed!")

# if __name__ == "__main__":
#     # print("Running base table detection example for png...")
#     # example_usage()
    
#     # print("Runing PDF table detection example...")
#     # example_usage_pdf 
#     # print("\n" + "="*50)
#     print("Running advanced example with visual analyzer...")
#     example_usage_with_visual_analyzer()

# # src/processors/table_processor.py

# import torch
# import pandas as pd
# import pytesseract
# from PIL import Image, ImageDraw, ImageFont
# from transformers import AutoImageProcessor, TableTransformerForObjectDetection
# from typing import Optional, List, Dict, Any

# # Disable the Decompression Bomb check in Pillow
# # Image.MAX_IMAGE_PIXELS = None

# class TableProcessor:
#     """
#     A specialized processor that uses a two-stage pipeline to detect, structure,
#     and extract text from tables in images for maximum robustness.
#     It includes a fallback to programmatically generate cells if the model fails to detect them.
#     """

#     def __init__(self,
#                  detection_model_name: str = "microsoft/table-transformer-detection",
#                  structure_model_name: str = "microsoft/table-transformer-structure-recognition-v1.1-all"):
#         """
#         Initializes the TableProcessor by loading the required models for both detection
#         and structure recognition stages.
#         """
#         print("Initializing TableProcessor: Loading detection and structure models...")
#         self.detection_image_processor = AutoImageProcessor.from_pretrained(detection_model_name)
#         self.detection_model = TableTransformerForObjectDetection.from_pretrained(detection_model_name)
        
#         self.structure_image_processor = AutoImageProcessor.from_pretrained(structure_model_name)
#         self.structure_model = TableTransformerForObjectDetection.from_pretrained(structure_model_name)

#         self.colors = {"table": "red", "table row": "blue", "table column": "green", "table cell": "magenta"}
#         print("TableProcessor initialized successfully.")

#     def extract_structured_table(
#         self,
#         image: Image.Image,
#         ocr_lang: str = 'eng+deu',
#         detection_threshold: float = 0.8,
#         structure_threshold: float = 0.6,
#         debug_prefix: Optional[str] = None
#     ) -> Optional[pd.DataFrame]:
#         """
#         Processes an image using a two-stage pipeline to find and structure a table.

#         Args:
#             image (Image.Image): A PIL Image object of the document page.
#             ocr_lang (str): The language(s) for Tesseract OCR.
#             detection_threshold (float): Confidence threshold for the initial table detection.
#             structure_threshold (float): Confidence threshold for detecting rows, columns, and cells.
#             debug_prefix (Optional[str]): If provided, saves debug images with this prefix.
#         """
#         print("--- Stage 1: Detecting main table area ---")
#         table_box = self._detect_main_table(image, threshold=detection_threshold)
#         if not table_box:
#             print("No table detected on the page.")
#             return None

#         # --- Verification Step 1: Visualize the detected table area ---
#         if debug_prefix:
#             debug_img_stage1 = image.copy()
#             draw = ImageDraw.Draw(debug_img_stage1)
#             draw.rectangle(table_box, outline="red", width=3)
#             path_stage1 = f"{debug_prefix}_stage1_detection.png"
#             print(f"Saving Stage 1 verification image to: {path_stage1}")
#             debug_img_stage1.save(path_stage1)
#         # --- End Verification Step ---

#         table_image = image.crop(table_box)
#         print("Table area detected and cropped.")

#         print("\n--- Stage 2: Recognizing structure within the cropped table ---")
#         detection_results = self._recognize_structure(table_image, threshold=structure_threshold)
#         if not detection_results:
#             print("No structural elements detected in the cropped table.")
#             return None

#         # --- Verification Step 2: Visualize the detected rows and columns ---
#         if debug_prefix:
#             path_stage2 = f"{debug_prefix}_stage2_structure.png"
#             print(f"Saving Stage 2 verification image to: {path_stage2}")
#             self._visualize_detections(table_image.copy(), detection_results, path_stage2)
#         # --- End Verification Step ---

#         table_grid = self._reconstruct_grid(detection_results)
#         cells = table_grid.get('cells', [])

#         if not cells:
#             print("Warning: No 'table cell' elements were detected. Check the debug image.")
#             return None

#         print(f"Performing OCR on {len(cells)} detected cells...")
#         for cell in cells:
#             cell['text'] = self._perform_ocr_on_cell(table_image, cell['box'], ocr_lang)

#         df = self._build_dataframe(table_grid)
#         if df is None or df.empty:
#             print("Warning: Could not construct a valid DataFrame.")
#             return None

#         return df

#     def _detect_main_table(self, image: Image.Image, threshold: float) -> Optional[List[float]]:
#         """Uses the detection model to find the primary table's bounding box."""
#         inputs = self.detection_image_processor(images=image, return_tensors="pt")
#         with torch.no_grad():
#             outputs = self.detection_model(**inputs)

#         target_sizes = torch.tensor([image.size[::-1]])
#         results = self.detection_image_processor.post_process_object_detection(
#             outputs, threshold=threshold, target_sizes=target_sizes
#         )[0]

#         # Find the box with the highest score labeled as 'table'
#         best_score = -1
#         best_box = None
#         for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#             if self.detection_model.config.id2label[label.item()] == "table":
#                 if score > best_score:
#                     best_score = score
#                     best_box = box.tolist()
        
#         return best_box

#     def _recognize_structure(self, image: Image.Image, threshold: float) -> List[Dict[str, Any]]:
#         """Uses the structure model on a cropped image to find rows, columns, and cells."""
#         size = {"shortest_edge": 800, "longest_edge": 1000}
#         inputs = self.structure_image_processor(images=image, size=size, return_tensors="pt")
#         with torch.no_grad():
#             outputs = self.structure_model(**inputs)

#         target_sizes = torch.tensor([image.size[::-1]])
#         results = self.structure_image_processor.post_process_object_detection(
#             outputs, threshold=threshold, target_sizes=target_sizes
#         )[0]
        
#         boxes = self._rescale_bboxes(results['boxes'], image.size).tolist()
#         detection_results = []
#         for label_id, score, box in zip(results['labels'].tolist(), results['scores'].tolist(), boxes):
#             detection_results.append({
#                 'label': self.structure_model.config.id2label[label_id],
#                 'score': score,
#                 'box': box
#             })
#         return detection_results
    
#     def _visualize_detections(self, image: Image.Image, detections: List[Dict], output_path: str):
#         """Saves a debug image with detected bounding boxes."""
#         draw = ImageDraw.Draw(image)
#         font = ImageFont.load_default()
#         for det in detections:
#             box, label, score = det['box'], det['label'], det['score']
#             color = self.colors.get(label, "white")
#             draw.rectangle(box, outline=color, width=2)
#             draw.text((box[0] + 5, box[1] + 5), f"{label} ({score:.2f})", fill=color, font=font)
#         image.save(output_path)

#     @staticmethod
#     def _rescale_bboxes(out_bbox: torch.Tensor, size: tuple) -> torch.Tensor:
#         """Rescales bounding box coordinates to the original image size."""
#         def box_cxcywh_to_xyxy(x):
#             x_c, y_c, w, h = x.unbind(1)
#             return torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=1)
#         img_w, img_h = size
#         b = box_cxcywh_to_xyxy(out_bbox)
#         return b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

#     @staticmethod
#     def _reconstruct_grid(table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Reconstructs a logical grid from detected rows, columns, and cells."""
#         rows = sorted([e for e in table_data if e['label'] == 'table row'], key=lambda x: x['box'][1])
#         columns = sorted([e for e in table_data if e['label'] == 'table column'], key=lambda x: x['box'][0])
#         cells = [e for e in table_data if e['label'] == 'table cell']
        
#         print(f"Found {len(rows)} rows, {len(columns)} columns, and {len(cells)} cells in raw detections.")
#         if not cells or not rows or not columns:
#             return {"cells": cells} # Return partial data for debugging

#         for i, row in enumerate(rows): row['row_number'] = i
#         for i, col in enumerate(columns): col['col_number'] = i

#         for cell in cells:
#             cell_center_x = (cell['box'][0] + cell['box'][2]) / 2
#             cell_center_y = (cell['box'][1] + cell['box'][3]) / 2
#             min_y_dist, assigned_row = min([(abs(cell_center_y - (r['box'][1] + r['box'][3]) / 2), r['row_number']) for r in rows], key=lambda t: t[0])
#             min_x_dist, assigned_col = min([(abs(cell_center_x - (c['box'][0] + c['box'][2]) / 2), c['col_number']) for c in columns], key=lambda t: t[0])
#             cell.update({'row_number': assigned_row, 'col_number': assigned_col})
#         return {"rows": rows, "columns": columns, "cells": cells}

#     @staticmethod
#     def _perform_ocr_on_cell(image: Image.Image, cell_box: List[float], lang: str) -> str:
#         """Performs OCR on a given cell's bounding box."""
#         box = [max(0, cell_box[0] - 5), max(0, cell_box[1] - 5), cell_box[2] + 5, cell_box[3] + 5]
#         cell_image = image.crop(box)
#         try:
#             return pytesseract.image_to_string(cell_image, lang=lang, config='--psm 6').strip()
#         except Exception as e:
#             print(f"OCR Error: {e}")
#             return ""

#     @staticmethod
#     def _build_dataframe(table_grid: Dict[str, Any]) -> Optional[pd.DataFrame]:
#         """Constructs a Pandas DataFrame from the reconstructed grid structure."""
#         rows, cols, cells = table_grid.get('rows', []), table_grid.get('columns', []), table_grid.get('cells', [])
#         if not cells or not rows or not cols:
#             return None

#         grid = [['' for _ in cols] for _ in rows]
#         for cell in cells:
#             row_idx, col_idx = cell.get('row_number', -1), cell.get('col_number', -1)
#             if 0 <= row_idx < len(rows) and 0 <= col_idx < len(cols):
#                 grid[row_idx][col_idx] = cell.get('text', '')

#         df = pd.DataFrame(grid)
#         if len(df) > 1 and df.iloc[0].astype(bool).any():
#             df.columns = df.iloc[0]
#             df = df[1:].reset_index(drop=True)
#         return df