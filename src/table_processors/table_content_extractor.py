from typing import List, Dict, Any, Optional
from ..pydantic_models.table_models import TableStructure, TableCell, TableData
from ..pydantic_models.semantic_table_models import SemanticTableExtraction
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class TableContentExtractor:
    """Enhanced table content extractor using GPT-4o with structured output"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4096
        )
        
        # Create structured output analyzer
        self.semantic_analyzer = self.llm.with_structured_output(SemanticTableExtraction)
    
    def extract_table_content(self, image_path: str, bbox: Dict[str, float], 
                            page_number: int = 1) -> TableData:
        """
        Extract table content with structured output for reliable results
        
        Args:
            image_path: Path to the document image
            bbox: Relative bounding box coordinates (0.0-1.0)
            page_number: Page number for metadata
            
        Returns:
            TableData: Structured table data with semantic relationships
        """
        logger.info(f"Extracting table content from {image_path} using structured output")
        
        # Crop table region from image
        table_image_base64 = self._crop_and_encode_table(image_path, bbox)
        
        # Create structured extraction prompt
        prompt = self._create_structured_extraction_prompt()
        
        # Call GPT-4o with structured output
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{table_image_base64}"}}
        ])
        
        try:
            # Get structured result directly as Pydantic object
            semantic_result: SemanticTableExtraction = self.semantic_analyzer.invoke([message])
            
            logger.info(f"Successfully extracted {len(semantic_result.tables)} table(s) with confidence {semantic_result.extraction_metadata.confidence:.2f}")
            
            # Convert to TableData format for compatibility
            return self._convert_semantic_to_table_data(semantic_result, bbox, page_number)
            
        except Exception as e:
            logger.error(f"Structured table extraction failed: {e}")
            # Simplified fallback
            return self._create_simple_fallback(bbox, page_number)
    
    def _create_structured_extraction_prompt(self) -> str:
        """Create prompt optimized for structured output"""
        return """
Analyze this table image and extract its content using structured semantic understanding.

IMPORTANT: You must return data in the exact format specified by the SemanticTableExtraction schema.

ANALYSIS TASKS:

1. **IDENTIFY TABLE TYPE AND STRUCTURE**:
   - Count exact rows and columns
   - Determine if headers exist (row/column headers)
   - Classify table type: specification_table, data_table, comparison_table, parameter_table, or other

2. **EXTRACT SEMANTIC RELATIONSHIPS**:
   For each data row, create a relationship with:
   - row_header: The label/name for this row
   - values: Dictionary mapping column headers to cell values
   - single_value: If it's a simple 2-column table (parameter → value), put the value here
   - row_notes: Any additional notes about this row

3. **HANDLE DIFFERENT TABLE FORMATS**:
   
   **For Parameter-Value tables** (2 columns):
   ```
   Parameter | Value
   durchfluss | 20-50 l/min
   ```
   Output as:
   ```
   {
     "row_header": "durchfluss",
     "single_value": "20-50 l/min",
     "values": {}
   }
   ```
   
   **For Multi-column tables**:
   ```
   Parameter | Min | Max | Unit
   durchfluss | 20 | 50 | l/min
   ```
   Output as:
   ```
   {
     "row_header": "durchfluss", 
     "values": {"Min": "20", "Max": "50", "Unit": "l/min"},
     "single_value": null
   }
   ```

   **For Description tables**:
   ```
   Parameter | Description
   Stellort | Kesselhaus
   ```
   Output as:
   ```
   {
     "row_header": "Stellort",
     "single_value": "Kesselhaus", 
     "values": {},
     "row_notes": null
   }
   ```

4. **QUALITY ASSESSMENT**:
   - confidence: 0.0-1.0 based on text clarity
   - detected_language: "de", "en", or "mixed"  
   - complexity_level: "simple", "moderate", or "complex"

CRITICAL REQUIREMENTS:
- ALWAYS provide the "values" field (use empty dict {} if no column values)
- For German technical terms, preserve original text
- For units, keep them with the values (20 l/min, 5.2 bar, etc.)
- If table structure is unclear, simplify to parameter-value pairs using single_value
- Ensure ALL required fields are present in the output
- If you see parameter names like "Stellort", "MSR-Aufgabe", etc., treat as parameter-description table

EXAMPLE OUTPUT STRUCTURE:
```json
{
  "table_summary": {
    "total_tables": 1,
    "main_topic": "Technical specifications",
    "document_type": "datasheet"
  },
  "tables": [{
    "table_id": "table_1",
    "title": "Parameter Specifications", 
    "table_type": "specification_table",
    "structure": {
      "rows": 3,
      "columns": 2,
      "has_row_headers": true,
      "has_column_headers": true,
      "header_levels": 1
    },
    "headers": {
      "row_headers": ["Parameter1", "Parameter2"],
      "column_headers": ["Parameter", "Value"]
    },
    "data_relationships": [
      {
        "row_header": "Stellort",
        "values": {},
        "single_value": "Kesselhaus",
        "row_notes": null
      }
    ],
    "notes": "Technical parameter table"
  }],
  "extraction_metadata": {
    "confidence": 0.9,
    "detected_language": "de", 
    "data_types": ["text"],
    "complexity_level": "simple"
  }
}
```

Focus on creating accurate data relationships that match the actual table structure.
"""
    
    def _crop_and_encode_table(self, image_path: str, bbox: Dict[str, float]) -> str:
        """Crop table region and encode to base64"""
        # Load image
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        # Convert relative coordinates to absolute pixels
        abs_x = int(bbox['x'] * img_width)
        abs_y = int(bbox['y'] * img_height) 
        abs_width = int(bbox['width'] * img_width)
        abs_height = int(bbox['height'] * img_height)
        
        # Crop table region
        table_image = image.crop((abs_x, abs_y, abs_x + abs_width, abs_y + abs_height))
        
        # Convert to base64
        import io
        buffer = io.BytesIO()
        table_image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _convert_semantic_to_table_data(self, semantic_result: SemanticTableExtraction, 
                                      bbox: Dict[str, float], 
                                      page_number: int) -> TableData:
        """Convert SemanticTableExtraction to TableData for compatibility"""
        
        # Use the first table (can be extended for multi-table support)
        main_table = semantic_result.tables[0]
        
        # Create cell objects from semantic relationships
        cells = []
        
        # Add column headers
        column_headers = main_table.headers.get("column_headers", [])
        for col_idx, header in enumerate(column_headers):
            cell = TableCell(
                id=f"header_col_{col_idx}",
                row=0,
                col=col_idx + 1,  # Offset for row header column
                content=header,
                cell_type="header",
                confidence=0.9
            )
            cells.append(cell)
        
        # Add data from relationships
        for row_idx, relationship in enumerate(main_table.data_relationships, start=1):
            row_header = relationship.row_header
            values = relationship.values
            single_value = relationship.single_value
            
            # Add row header cell
            row_cell = TableCell(
                id=f"header_row_{row_idx}",
                row=row_idx,
                col=0,
                content=row_header,
                cell_type="header", 
                confidence=0.9
            )
            cells.append(row_cell)
            
            # Handle different value formats
            if single_value and not values:
                # Simple parameter → value table
                data_cell = TableCell(
                    id=f"data_{row_idx}_1",
                    row=row_idx,
                    col=1,
                    content=single_value,
                    cell_type="data",
                    confidence=0.85
                )
                cells.append(data_cell)
            else:
                # Multi-column table with values dict
                for col_idx, (col_header, value) in enumerate(values.items(), start=1):
                    data_cell = TableCell(
                        id=f"data_{row_idx}_{col_idx}",
                        row=row_idx,
                        col=col_idx,
                        content=str(value) if value else "",
                        cell_type="data",
                        confidence=0.85
                    )
                    cells.append(data_cell)
        
        # Create table structure
        structure = TableStructure(
            id=main_table.table_id,
            rows=main_table.structure.rows,
            cols=main_table.structure.columns,
            cells=cells,
            headers=column_headers,
            caption=main_table.title,
            page_number=page_number,
            bbox=bbox,
            extraction_method="gpt4v_visual",  # Use valid enum value
            structure_confidence=semantic_result.extraction_metadata.confidence,
            metadata={
                "semantic_extraction": True,
                "table_type": main_table.table_type,
                "detected_language": semantic_result.extraction_metadata.detected_language,
                "data_types": semantic_result.extraction_metadata.data_types,
                "complexity_level": semantic_result.extraction_metadata.complexity_level,
                "total_tables": semantic_result.table_summary.total_tables,
                "document_type": semantic_result.table_summary.document_type,
                "notes": main_table.notes
            }
        )
        
        # Create data rows from relationships
        data_rows = []
        for relationship in main_table.data_relationships:
            row_data = {"row_header": relationship.row_header}
            
            # Handle different value formats
            if relationship.single_value and not relationship.values:
                # Simple parameter → value format
                row_data["value"] = relationship.single_value
            else:
                # Multi-column format
                row_data.update(relationship.values)
            
            # Add notes if present
            if relationship.row_notes:
                row_data["notes"] = relationship.row_notes
                
            data_rows.append(row_data)
        
        return TableData(
            structure=structure,
            data_rows=data_rows,
            metadata={
                "extraction_approach": "structured_semantic",
                "bbox": bbox,
                "page_number": page_number,
                "semantic_result": semantic_result.model_dump(),  # Store complete result
                "main_topic": semantic_result.table_summary.main_topic
            }
        )
    
    def _create_simple_fallback(self, bbox: Dict[str, float], page_number: int) -> TableData:
        """Create minimal fallback TableData when structured extraction fails"""
        logger.warning("Creating simple fallback table data")
        
        # Create minimal valid structure
        headers = ["Parameter", "Value"]
        data_rows = [{"Parameter": "Extraction Failed", "Value": "Unable to parse table"}]
        
        cells = [
            TableCell(
                id="fallback_header_0", row=0, col=0, 
                content="Parameter", cell_type="header", confidence=0.3
            ),
            TableCell(
                id="fallback_header_1", row=0, col=1,
                content="Value", cell_type="header", confidence=0.3
            ),
            TableCell(
                id="fallback_data_1_0", row=1, col=0,
                content="Extraction Failed", cell_type="data", confidence=0.3
            ),
            TableCell(
                id="fallback_data_1_1", row=1, col=1,
                content="Unable to parse table", cell_type="data", confidence=0.3
            )
        ]
        
        structure = TableStructure(
            id="fallback_table",
            rows=2,
            cols=2,
            cells=cells,
            headers=headers,
            page_number=page_number,
            bbox=bbox,
            extraction_method="manual",
            structure_confidence=0.3,
            metadata={"fallback_used": True, "structured_extraction_failed": True}
        )
        
        return TableData(
            structure=structure,
            data_rows=data_rows,
            metadata={
                "extraction_approach": "simple_fallback",
                "bbox": bbox,
                "page_number": page_number
            }
        )
    
    def _create_semantic_table_data(self, table_json: Dict[str, Any], 
                                  bbox: Dict[str, float], 
                                  page_number: int) -> TableData:
        """Legacy method - now handled by _convert_semantic_to_table_data"""
        logger.warning("_create_semantic_table_data is deprecated - use structured output instead")
        return self._create_simple_fallback(bbox, page_number)
    
    def _extract_with_fallback(self, response_text: str, 
                              bbox: Dict[str, float],
                              page_number: int) -> TableData:
        """Legacy fallback method - now handled by _create_simple_fallback"""
        logger.warning("_extract_with_fallback is deprecated - use structured output instead")
        return self._create_simple_fallback(bbox, page_number)
    
    def extract_multiple_tables(self, image_path: str, 
                               table_regions: List[Dict[str, Any]]) -> List[TableData]:
        """
        Extract content from multiple table regions
        
        Args:
            image_path: Path to the document image
            table_regions: List of table region dictionaries with bbox info
            
        Returns:
            List of TableData objects
        """
        results = []
        
        for i, region in enumerate(table_regions):
            try:
                bbox = region.get('bbox', region)  # Handle both formats
                page_number = region.get('page_number', 1)
                
                table_data = self.extract_table_content(
                    image_path, bbox, page_number
                )
                
                # Add region identifier
                table_data.metadata['region_index'] = i
                table_data.metadata['region_id'] = region.get('id', f'table_{i}')
                
                results.append(table_data)
                
            except Exception as e:
                logger.error(f"Failed to extract table {i}: {e}")
                continue
        
        return results


# Usage example
def example_usage():
    """Example of using the enhanced TableContentExtractor with PDF input"""
    import os
    from ..utils.image_utils import convert_pdf_to_images
    from ..core.region_detector import RegionDetector
    from ..pydantic_models.enums import RegionType
    
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    pdf_path = "data/inputs/sample_Datenblatt.pdf"  # Path to your PDF file
    output_dir = "data/outputs/temp_images"  # Temporary directory for images
    
    # Initialize components
    extractor = TableContentExtractor(api_key)
    region_detector = RegionDetector(api_key)
    
    try:
        print("=== PDF to Image Conversion ===")
        # Convert PDF to images
        images = convert_pdf_to_images(
            pdf_path=pdf_path,
            # output_dir=output_dir,
            dpi=300,  # High resolution for better OCR
        )
        
        print(f"Converted PDF to {len(images)} image(s)")
        
        # Process each page
        all_results = []
        
        for page_num, image in enumerate(images, 1):
            # Handle both PIL Image objects and file paths
            if hasattr(image, 'save'):  # PIL Image object
                # Save PIL Image to temporary file
                import tempfile
                temp_path = os.path.join(output_dir, f"temp_page_{page_num}.png")
                os.makedirs(output_dir, exist_ok=True)
                image.save(temp_path)
                image_path = temp_path
            else:
                # Already a file path
                image_path = image
            print(f"\n=== Processing Page {page_num} ===")
            print(f"Image: {image_path}")
            
            # Step 1: Detect regions using RegionDetector
            print("Detecting document regions...")
            layout = region_detector.analyze_document(
                image_path=image_path,
                page_number=page_num
            )
            
            # Step 2: Find table regions
            table_regions = [r for r in layout.regions if r.type == RegionType.TABLE]
            print(f"Found {len(table_regions)} table region(s)")
            
            if not table_regions:
                print("No tables detected on this page")
                continue
            
            # Step 3: Extract content from each table
            page_results = []
            for i, table_region in enumerate(table_regions):
                print(f"\n--- Extracting Table {i+1} ---")
                print(f"Region ID: {table_region.id}")
                print(f"Confidence: {table_region.confidence:.2f}")
                print(f"Description: {table_region.content_description}")
                
                try:
                    # Convert bbox to dict format
                    bbox_dict = {
                        'x': table_region.bbox.x,
                        'y': table_region.bbox.y,
                        'width': table_region.bbox.width,
                        'height': table_region.bbox.height
                    }
                    
                    # Extract table content
                    result = extractor.extract_table_content(
                        image_path=image_path,
                        bbox=bbox_dict,
                        page_number=page_num
                    )
                    
                    # Display extraction results
                    print(f"Table ID: {result.structure.id}")
                    print(f"Dimensions: {result.structure.rows}x{result.structure.cols}")
                    print(f"Confidence: {result.structure.structure_confidence:.2f}")
                    print(f"Language: {result.structure.metadata.get('detected_language', 'unknown')}")
                    print(f"Table Type: {result.structure.metadata.get('table_type', 'unknown')}")
                    print(f"Complexity: {result.structure.metadata.get('complexity_level', 'unknown')}")
                    print(f"Document Type: {result.structure.metadata.get('document_type', 'unknown')}")
                    
                    # Show semantic data relationships
                    print("\n=== Data Relationships ===")
                    for row_idx, row in enumerate(result.data_rows[:3]):  # Show first 3 rows
                        row_header = row.get('row_header', f'Row {row_idx+1}')
                        print(f"  {row_header}:")
                        for col_header, value in row.items():
                            if col_header != 'row_header':
                                print(f"    {col_header} → {value}")
                    
                    if len(result.data_rows) > 3:
                        print(f"    ... and {len(result.data_rows) - 3} more rows")
                    
                    # Check for sub-tables from structured output
                    total_tables = result.structure.metadata.get('total_tables', 1)
                    if total_tables > 1:
                        print(f"\n=== Sub-tables Detected: {total_tables} ===")
                        semantic_result = result.metadata.get('semantic_result', {})
                        all_tables = semantic_result.get('tables', [])
                        for j, sub_table in enumerate(all_tables[:3]):  # Show first 3 sub-tables
                            print(f"  Sub-table {j+1}: {sub_table.get('title', 'Untitled')}")
                            print(f"    Type: {sub_table.get('table_type', 'unknown')}")
                            structure = sub_table.get('structure', {})
                            print(f"    Size: {structure.get('rows', '?')}x{structure.get('columns', '?')}")
                            print(f"    Relationships: {len(sub_table.get('data_relationships', []))}")
                    
                    # Show main topic
                    main_topic = result.metadata.get('main_topic')
                    if main_topic:
                        print(f"\n=== Main Topic ===")
                        print(f"  {main_topic}")
                    
                    page_results.append(result)
                    
                except Exception as e:
                    print(f"Failed to extract table {i+1}: {e}")
                    continue
            
            all_results.extend(page_results)
            print(f"\nPage {page_num} completed: {len(page_results)} table(s) extracted")
        
        # Summary
        print(f"\n=== Extraction Summary ===")
        print(f"Total pages processed: {len(images)}")
        print(f"Total tables extracted: {len(all_results)}")
        
        # Show overall statistics
        if all_results:
            avg_confidence = sum(r.structure.structure_confidence for r in all_results) / len(all_results)
            print(f"Average extraction confidence: {avg_confidence:.2f}")
            
            languages = set()
            table_types = set()
            complexity_levels = set()
            document_types = set()
            
            for result in all_results:
                metadata = result.structure.metadata
                
                lang = metadata.get('detected_language')
                if lang:
                    languages.add(lang)
                    
                table_type = metadata.get('table_type')
                if table_type:
                    table_types.add(table_type)
                    
                complexity = metadata.get('complexity_level')
                if complexity:
                    complexity_levels.add(complexity)
                    
                doc_type = metadata.get('document_type')
                if doc_type:
                    document_types.add(doc_type)
            
            print(f"Detected languages: {', '.join(languages) if languages else 'unknown'}")
            print(f"Table types found: {', '.join(table_types) if table_types else 'unknown'}")
            print(f"Complexity levels: {', '.join(complexity_levels) if complexity_levels else 'unknown'}")
            print(f"Document types: {', '.join(document_types) if document_types else 'unknown'}")
            
            # Show extraction method usage
            extraction_methods = set(r.structure.extraction_method for r in all_results)
            print(f"Extraction methods used: {', '.join(extraction_methods)}")
            
            # Count successful vs fallback extractions
            structured_count = sum(1 for r in all_results 
                                 if r.metadata.get('extraction_approach') == 'structured_semantic')
            fallback_count = len(all_results) - structured_count
            print(f"Successful structured extractions: {structured_count}")
            print(f"Fallback extractions: {fallback_count}")
        
        # Optional: Clean up temporary images
        print(f"\nTemporary images saved in: {output_dir}")
        print("To clean up: remove the temp_images directory")
        
        return all_results
        
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please ensure the PDF file exists and path is correct")
        return []
    
    except Exception as e:
        print(f"Processing failed: {e}")
        return []


def example_usage_simple():
    """Simplified example for single image input"""
    
    # Initialize extractor
    extractor = TableContentExtractor("your-openai-api-key")
    
    # Example bbox (relative coordinates)
    bbox = {
        'x': 0.1,      # 10% from left
        'y': 0.2,      # 20% from top  
        'width': 0.8,  # 80% of image width
        'height': 0.6  # 60% of image height
    }
    
    try:
        # Extract table content from image
        result = extractor.extract_table_content(
            image_path="document_page.jpg",
            bbox=bbox,
            page_number=1
        )
        
        # Display results
        print("=== Table Extraction Results ===")
        print(f"Table ID: {result.structure.id}")
        print(f"Dimensions: {result.structure.rows}x{result.structure.cols}")
        print(f"Confidence: {result.structure.structure_confidence:.2f}")
        
        print("\n=== Data Relationships ===")
        for row in result.data_rows:
            print(f"Row: {row}")
        
    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    # Run the full PDF processing example
    results = example_usage()
    
    # Uncomment below for simple image example
    # example_usage_simple()