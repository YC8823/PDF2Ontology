# ==================== src/analyzers/region_detector.py ====================
"""
Region detection and analysis using GPT-4o vision capabilities
Updated to work with new Pydantic models and DocumentAnalyzer
"""

import os
import base64
import json
import logging
from typing import Dict, Optional, List
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import new Pydantic models
from ..pydantic_models.region_models import DocumentLayout, DocumentRegion, BoundingBox
from ..pydantic_models.table_models import TableStructure
from ..pydantic_models.enums import RegionType

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

class RegionDetector:
    """Detects and analyzes regions in document images using GPT-4o with structured output"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            max_tokens=4096,
            temperature=0
        )

        # Create structured output LLM chains for different tasks
        self.layout_analyzer = self.llm.with_structured_output(DocumentLayout)
        self.table_analyzer = self.llm.with_structured_output(TableStructure)
        
    def analyze_document(self, image_path: str, page_number: int = 1, 
                        document_id: str = None) -> DocumentLayout:
        """
        Analyze document layout and detect regions
        
        Args:
            image_path: Path to the image file
            page_number: Page number (1-indexed)
            document_id: Optional document identifier
            
        Returns:
            DocumentLayout: Layout analysis result with detected regions
        """
        
        logger.info(f"Starting region detection for: {image_path}")
        
        # Get image metadata
        with Image.open(image_path) as img:
            image_dimensions = {"width": img.width, "height": img.height}
        
        # Encode image for API
        base64_image = self._encode_image(image_path)
        
        # Create analysis prompt
        prompt = self._create_region_detection_prompt(image_dimensions, page_number)
        
        # Analyze with structured output
        try:
            result = self._analyze_layout_structured(
                base64_image, prompt, document_id, image_dimensions, page_number
            )
            logger.info(f"Detected {len(result.regions)} regions using structured output")
            return result
        
        except Exception as e:
            logger.error(f"Structured region detection failed: {str(e)}")
            raise
    
    def detect_regions(self, image_path: str, page_number: int = 1, 
                      document_id: str = None) -> DocumentLayout:
        """
        Legacy method name for backward compatibility
        Delegates to analyze_document
        """
        return self.analyze_document(image_path, page_number, document_id)
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_region_detection_prompt(self, image_dimensions: Dict[str, int], page_number: int) -> str:
        """Create prompt for region detection with structured output"""
        
        width, height = image_dimensions["width"], image_dimensions["height"]
        
        prompt = f"""
Analyze this document image and identify ALL distinct regions with precise bounding box coordinates.

Image specifications:
- Dimensions: {width} × {height} pixels
- Coordinate system: Origin (0,0) at top-left, X increases right, Y increases down
- Page number: {page_number}

Detection requirements:
1. Identify every distinct content region in the document
2. Classify each region using these types:
   - TEXT: Regular paragraph text
   - TABLE: Tabular data with rows/columns
   - IMAGE: Pictures, diagrams, charts
   - HEADER: Page headers
   - FOOTER: Page footers  
   - TITLE: Document or section titles
   - CAPTION: Image/table captions
   - SIDEBAR: Side panels or callouts
   - OTHER: Any other content type

3. Provide exact bounding box coordinates in relative format (0.0 to 1.0)
4. Assign confidence scores based on region clarity and boundaries (0.0 to 1.0)
5. Generate unique region IDs
6. Describe content briefly but accurately
7. Determine reading order where applicable (1, 2, 3, ...)

Critical constraints:
- Use relative coordinates: x, y, width, height all between 0.0 and 1.0
- x = left_pixel / {width}, y = top_pixel / {height}
- width = region_width / {width}, height = region_height / {height}
- Confidence scores must be between 0.0 and 1.0
- Reading order should reflect natural document flow

Pay special attention to:
- Multi-column text layouts
- Table boundaries and structure  
- Image regions and associated captions
- Headers, footers, and title areas
- Sidebar or callout content
- Clear separation between regions

The response must conform exactly to the DocumentLayout schema with relative coordinates.
"""
        return prompt
    
    def _analyze_layout_structured(self, base64_image: str, prompt: str, 
                                 document_id: Optional[str], 
                                 image_dimensions: Dict[str, int],
                                 page_number: int) -> DocumentLayout:
        """Perform structured layout analysis using LLM"""
        
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
            # Get structured result directly as Pydantic object
            result: DocumentLayout = self.layout_analyzer.invoke([message])
            
            # Create a new result with additional fields using model_copy and update
            # Note: Cannot directly assign fields not in the original model
            result_dict = result.model_dump()
            
            # Add additional metadata
            result_dict.update({
                "page_number": page_number,
                "image_dimensions": image_dimensions,
                "document_metadata": {
                    "document_id": document_id or f"doc_{page_number}",
                    "analysis_timestamp": "timestamp_placeholder",
                    "model_used": "gpt-4o"
                }
            })
            
            # Validate and update all regions
            validated_regions = []
            for i, region in enumerate(result.regions):
                try:
                    # Validate bounding box
                    self._validate_relative_bbox(region.bbox)
                    
                    # Ensure region has page number
                    region.page_number = page_number
                    
                    # Generate ID if missing
                    if not region.id:
                        region.id = f"region_{page_number}_{i+1}"
                    
                    validated_regions.append(region)
                    
                except ValueError as e:
                    logger.warning(f"Removing invalid region {region.id or i}: {str(e)}")
                    continue
            
            # Create new result with validated regions
            result = result.model_copy(update={"regions": validated_regions})
            
            # Update result with additional computed fields if they exist in the model
            if hasattr(result, 'total_regions'):
                result.total_regions = len(validated_regions)
            if hasattr(result, 'region_types'):
                result.region_types = list(set(r.type for r in validated_regions))
            
            return result
            
        except Exception as e:
            logger.error(f"Structured layout analysis failed: {str(e)}")
            raise ValueError(f"Layout analysis error: {str(e)}")
    
    def _validate_relative_bbox(self, bbox: BoundingBox):
        """Validate relative bounding box coordinates (0.0 to 1.0)"""
        
        if not (0.0 <= bbox.x <= 1.0):
            raise ValueError(f"Invalid x coordinate: {bbox.x} (must be 0.0-1.0)")
        
        if not (0.0 <= bbox.y <= 1.0):
            raise ValueError(f"Invalid y coordinate: {bbox.y} (must be 0.0-1.0)")
        
        if not (0.0 < bbox.width <= 1.0):
            raise ValueError(f"Invalid width: {bbox.width} (must be 0.0-1.0)")
        
        if not (0.0 < bbox.height <= 1.0):
            raise ValueError(f"Invalid height: {bbox.height} (must be 0.0-1.0)")
        
        if bbox.x + bbox.width > 1.0:
            raise ValueError(f"Bbox extends beyond image: x + width = {bbox.x + bbox.width} > 1.0")
        
        if bbox.y + bbox.height > 1.0:
            raise ValueError(f"Bbox extends beyond image: y + height = {bbox.y + bbox.height} > 1.0")
    
    def analyze_table_structure(self, image_path: str, table_region: DocumentRegion) -> TableStructure:
        """
        Analyze table structure using structured output
        
        Args:
            image_path: Path to the source image
            table_region: DocumentRegion representing the table
            
        Returns:
            TableStructure: Detailed table structure analysis
        """
        
        # Load image and convert relative coordinates to absolute
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        bbox = table_region.bbox
        
        # Convert relative coordinates to absolute pixels
        abs_x = int(bbox.x * img_width)
        abs_y = int(bbox.y * img_height)
        abs_width = int(bbox.width * img_width)
        abs_height = int(bbox.height * img_height)
        
        # Crop table region from image
        table_image = image.crop((abs_x, abs_y, abs_x + abs_width, abs_y + abs_height))
        
        # Convert cropped image to base64
        import io
        buffer = io.BytesIO()
        table_image.save(buffer, format='PNG')
        table_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create table analysis prompt
        prompt = f"""
Analyze this table image and extract its complete structure and content.

Task requirements:
1. Determine exact number of rows and columns
2. Extract text content from each cell
3. Identify header rows/columns if present
4. Handle merged cells (rowspan/colspan > 1)
5. Provide accurate cell data
6. Assess extraction confidence

Table information:
- Source region ID: {table_region.id}
- Original description: {table_region.content_description}
- Page number: {table_region.page_number}

Quality assessment:
- Assign confidence scores based on text clarity and cell boundary detection
- Higher scores for clear, well-structured tables
- Lower scores for blurry or complex tables

The response must conform to the TableStructure schema with complete cell data.
"""
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{table_base64}"}
                }
            ]
        )
        
        try:
            # Get structured table result
            table_result: TableStructure = self.table_analyzer.invoke([message])
            
            # Update table metadata to match source region
            if not table_result.id:
                table_result.id = table_region.id
            
            table_result.page_number = table_region.page_number
            table_result.bbox = table_region.bbox  # Keep relative coordinates
            
            # Update metadata
            if not table_result.metadata:
                table_result.metadata = {}
            
            table_result.metadata.update({
                "source_region_id": table_region.id,
                "original_description": table_region.content_description,
                "crop_coordinates": {
                    "abs_x": abs_x,
                    "abs_y": abs_y, 
                    "abs_width": abs_width,
                    "abs_height": abs_height
                }
            })
            
            return table_result
            
        except Exception as e:
            logger.error(f"Table structure analysis failed: {str(e)}")
            raise ValueError(f"Table analysis error: {str(e)}")
    
    def extract_text_from_regions(self, image_path: str, 
                                 text_regions: List[DocumentRegion]) -> List[Dict[str, any]]:
        """
        Extract text content from specified regions
        
        Args:
            image_path: Path to the source image
            text_regions: List of text-based regions to extract from
            
        Returns:
            List of extracted text data
        """
        
        base64_image = self._encode_image(image_path)
        
        # Filter for text-based regions
        valid_text_types = {RegionType.TEXT, RegionType.TITLE, RegionType.HEADER, 
                           RegionType.FOOTER, RegionType.CAPTION}
        text_regions = [r for r in text_regions if r.type in valid_text_types]
        
        if not text_regions:
            logger.warning("No valid text regions provided for extraction")
            return []
        
        # Create region descriptions for prompt
        region_descriptions = []
        for region in text_regions:
            desc = f"- {region.id} ({region.type.value}): {region.content_description}"
            region_descriptions.append(desc)
        
        prompt = f"""
Extract complete text content from all text-based regions in this document image.

Text regions to extract:
{chr(10).join(region_descriptions)}

Extraction requirements:
1. Extract complete, accurate text from each region
2. Preserve original formatting where possible
3. Maintain text structure and line breaks
4. Provide confidence assessment for each extraction
5. Return results as a list of extraction results

For each region, provide:
- region_id: The ID of the source region
- extracted_text: The complete text content
- confidence: Quality score (0.0 to 1.0)
- language: Detected primary language (if identifiable)
- word_count: Approximate number of words
- notes: Any extraction issues or observations

Quality assessment criteria:
- 1.0: Perfect, clear text extraction
- 0.8-0.9: High quality with minor issues
- 0.6-0.7: Good quality with some unclear characters
- 0.4-0.5: Moderate quality with several issues
- 0.0-0.3: Poor quality, significant extraction problems

Return as a JSON list of extraction results.
"""
        
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
            # Get text extraction result
            response = self.llm.invoke([message])
            
            # Parse JSON response
            try:
                extraction_results = json.loads(response.content)
                if isinstance(extraction_results, list):
                    return extraction_results
                else:
                    return [extraction_results]  # Wrap single result in list
            except json.JSONDecodeError:
                # Fallback: create basic extraction results
                logger.warning("Failed to parse JSON response, using fallback extraction")
                return self._create_fallback_extraction(text_regions, response.content)
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return []
    
    def _create_fallback_extraction(self, regions: List[DocumentRegion], 
                                  raw_text: str) -> List[Dict[str, any]]:
        """Create fallback extraction results when JSON parsing fails"""
        
        # Split text roughly by number of regions
        text_parts = raw_text.split('\n\n')
        results = []
        
        for i, region in enumerate(regions):
            text_part = text_parts[i] if i < len(text_parts) else "Extraction failed"
            
            result = {
                "region_id": region.id,
                "extracted_text": text_part.strip(),
                "confidence": 0.5,  # Low confidence for fallback
                "language": "unknown",
                "word_count": len(text_part.split()),
                "notes": "Fallback extraction used due to parsing error"
            }
            results.append(result)
        
        return results

# ==================== Usage Example ====================
def example_usage():
    """Example usage of the refactored RegionDetector"""
    
    # Initialize detector
    detector = RegionDetector("your-openai-api-key")  # Replace with actual API key
    
    # Analyze document layout
    image_path = "samples/page_3.png"
    
    try:
        # Detect regions
        layout = detector.analyze_document(image_path, page_number=1)
        
        print(f"=== Layout Analysis Results ===")
        print(f"Document ID: {layout.document_id}")
        print(f"Page: {layout.page_number}")
        print(f"Total regions: {layout.total_regions}")
        print(f"Region types: {[rt.value for rt in layout.region_types]}")
        
        # Show region details
        for region in layout.regions:
            print(f"\nRegion: {region.id}")
            print(f"  Type: {region.type.value}")
            print(f"  Confidence: {region.confidence:.2f}")
            print(f"  Bbox: ({region.bbox.x:.3f}, {region.bbox.y:.3f}, "
                  f"{region.bbox.width:.3f}, {region.bbox.height:.3f})")
            print(f"  Description: {region.content_description}")
        
        # Analyze tables if found
        table_regions = [r for r in layout.regions if r.type == RegionType.TABLE]
        if table_regions:
            print(f"\n=== Table Analysis ===")
            for table_region in table_regions:
                try:
                    table_structure = detector.analyze_table_structure(image_path, table_region)
                    print(f"Table {table_region.id}: {table_structure.rows}x{table_structure.cols}")
                    print(f"  Headers: {table_structure.headers}")
                    print(f"  Confidence: {table_structure.structure_confidence:.2f}")
                except Exception as e:
                    print(f"Table analysis failed for {table_region.id}: {e}")
        
        # Extract text from text regions
        text_regions = [r for r in layout.regions 
                       if r.type in {RegionType.TEXT, RegionType.TITLE, RegionType.HEADER}]
        if text_regions:
            print(f"\n=== Text Extraction ===")
            text_results = detector.extract_text_from_regions(image_path, text_regions)
            for result in text_results:
                print(f"Region {result['region_id']}: {result['word_count']} words, "
                      f"confidence: {result['confidence']:.2f}")
                print(f"  Text: {result['extracted_text'][:100]}...")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    example_usage()