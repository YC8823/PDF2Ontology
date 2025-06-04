"""
Region detection and analysis using GPT-4o vision capabilities
"""

import os
import sys
import base64
import json
import logging
from typing import Dict, Optional
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# from langchain_core.output_parsers import JsonOutputParser

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from .models import (
    DocumentAnalysisResult, 
    DocumentRegion,
    BoundingBox,
    RegionType
)

logger = logging.getLogger(__name__)

class RegionDetector:
    """Detects and analyzes regions in document images using GPT-4o"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            max_tokens=4096,
            temperature=0
        )

        #self.parser = JsonOutputParser(pydantic_object=DocumentAnalysisResult)

        # Create structured output LLM chains for different tasks

        self.layout_analyzer = self.llm.with_structured_output(DocumentAnalysisResult)
        #self.table_analyzer = self.llm.with_structured_output(TableStructure)
        #self.content_extractor = self.llm.with_structured_output(ContentExtractionResult)
        
    def detect_regions(self, image_path: str, page_number: int = 0, 
                      document_id: str = None) -> DocumentAnalysisResult:
        """Detect and analyze regions in document image"""
        
        logger.info(f"Starting region detection for: {image_path}")
        
        # Get image metadata
        with Image.open(image_path) as img:
            image_dimensions = {"width": img.width, "height": img.height}
        
        # Encode image for API
        base64_image = self._encode_image(image_path)
        
        # Create analysis prompt
        prompt = self._create_region_detection_prompt(image_dimensions, page_number)
        
        # Analyze with GPT-4o
        #analysis_result = self._analyze_with_llm(base64_image, prompt)
        # Analyze with structured output
        try:
            result = self._analyze_layout_structured(base64_image, prompt, document_id, image_dimensions)
            logger.info(f"Detected {len(result.regions)} regions using structured output")
            return result
        
        except Exception as e:
            logger.error(f"Structured region detection failed: {str(e)}")
            raise

        # Process and validate results
        # result = self._process_analysis_result(
        #     analysis_result, 
        #     document_id or image_path,
        #     image_dimensions,
        #     page_number
        # )
        
        # logger.info(f"Detected {len(result.regions)} regions")
        # return result
    
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
2. Classify each region as: text, table, image, header, footer, title, caption, or sidebar
3. Provide exact pixel coordinates for bounding boxes
4. Assign confidence scores based on region clarity and boundaries
5. Generate unique region IDs (region_1, region_2, etc.)
6. Describe content briefly but accurately
7. Determine reading order where applicable

Critical constraints:
- All coordinates must satisfy: 0 ≤ x < {width}, 0 ≤ y < {height}
- Bounding boxes must not extend beyond image boundaries
- Width and height must be positive integers
- Confidence scores must be between 0.0 and 1.0
- Reading order should reflect natural document flow

Pay special attention to:
- Multi-column text layouts
- Table boundaries and structure  
- Image regions and associated captions
- Headers, footers, and title areas
- Sidebar or callout content
- Overlapping or nested regions

The response must conform exactly to the DocumentAnalysisResult schema.
"""
        return prompt
    
    def _analyze_layout_structured(self, base64_image: str, prompt: str, 
                                 document_id: Optional[str], 
                                 image_dimensions: Dict[str, int]) -> DocumentAnalysisResult:
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
            result: DocumentAnalysisResult = self.layout_analyzer.invoke([message])
            
            # Post-process the result
            result.document_id = document_id or result.document_id
            result.image_dimensions = image_dimensions
            
            # Validate all bounding boxes
            validated_regions = []
            for region in result.regions:
                try:
                    self._validate_bbox(region.bbox, image_dimensions)
                    validated_regions.append(region)
                except ValueError as e:
                    logger.warning(f"Removing invalid region {region.region_id}: {str(e)}")
                    continue
            
            result.regions = validated_regions
            return result
            
        except Exception as e:
            logger.error(f"Structured layout analysis failed: {str(e)}")
            raise ValueError(f"Layout analysis error: {str(e)}")
        
    # def _analyze_with_llm(self, base64_image: str, prompt: str) -> Dict[str, Any]:
    #     """Send analysis request to GPT-4o"""
        
    #     message = HumanMessage(
    #         content=[
    #             {"type": "text", "text": prompt},
    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
    #             }
    #         ]
    #     )
        
    #     try:
    #         response = self.llm.invoke([message])
    #         result = json.loads(response.content)
    #         return result
            
    #     except json.JSONDecodeError as e:
    #         logger.error(f"Failed to parse LLM response: {str(e)}")
    #         logger.debug(f"Raw response: {response.content}")
    #         raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
    #     except Exception as e:
    #         logger.error(f"LLM analysis failed: {str(e)}")
    #         raise
    
    # def _process_analysis_result(self, raw_result: Dict[str, Any], 
    #                            document_id: str, image_dimensions: Dict[str, int],
    #                            page_number: int) -> DocumentAnalysisResult:
    #     """Process and validate analysis results"""
        
    #     # Extract regions and validate
    #     regions = []
    #     for i, region_data in enumerate(raw_result.get("regions", [])):
    #         try:
    #             # Validate and create BoundingBox
    #             bbox_data = region_data.get("bbox", {})
    #             bbox = BoundingBox(
    #                 x=bbox_data.get("x", 0),
    #                 y=bbox_data.get("y", 0), 
    #                 width=bbox_data.get("width", 0),
    #                 height=bbox_data.get("height", 0)
    #             )
                
    #             # Validate bbox is within image bounds
    #             self._validate_bbox(bbox, image_dimensions)
                
    #             # Create DocumentRegion
    #             region = DocumentRegion(
    #                 region_id=region_data.get("region_id", f"region_{i}"),
    #                 region_type=RegionType(region_data.get("region_type", "text")),
    #                 bbox=bbox,
    #                 confidence=region_data.get("confidence", 0.0),
    #                 content_description=region_data.get("content_description", ""),
    #                 page_number=page_number,
    #                 reading_order=region_data.get("reading_order"),
    #                 metadata=region_data.get("metadata", {})
    #             )
                
    #             regions.append(region)
                
    #         except Exception as e:
    #             logger.warning(f"Skipping invalid region {i}: {str(e)}")
    #             continue
        
    #     # Create final result
    #     result = DocumentAnalysisResult(
    #         document_id=document_id,
    #         regions=regions,
    #         page_layout=raw_result.get("page_layout", "Unknown layout"),
    #         total_pages=raw_result.get("total_pages", 1),
    #         image_dimensions=image_dimensions,
    #         processing_metadata=raw_result.get("processing_metadata", {})
    #     )
        
    #     return result
    
    def _validate_bbox(self, bbox: BoundingBox, image_dimensions: Dict[str, int]):
        """Validate bounding box coordinates"""
        
        width, height = image_dimensions["width"], image_dimensions["height"]
        
        if bbox.x < 0 or bbox.y < 0:
            raise ValueError(f"Negative coordinates: ({bbox.x}, {bbox.y})")
        
        if bbox.x + bbox.width > width:
            raise ValueError(f"Bbox extends beyond image width: {bbox.x + bbox.width} > {width}")
        
        if bbox.y + bbox.height > height:
            raise ValueError(f"Bbox extends beyond image height: {bbox.y + bbox.height} > {height}")
        
#     def analyze_table_structure(self, image_path: str, table_region: DocumentRegion) -> TableStructure:
#         """Analyze table structure using structured output"""
        
#         # Crop table region from image
#         image = Image.open(image_path)
#         bbox = table_region.bbox
#         table_image = image.crop((bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height))
        
#         # Convert cropped image to base64
#         import io
#         buffer = io.BytesIO()
#         table_image.save(buffer, format='PNG')
#         table_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
#         # Create table analysis prompt
#         prompt = f"""
# Analyze this table image and extract its complete structure and content.

# Task requirements:
# 1. Determine exact number of rows and columns
# 2. Extract text content from each cell
# 3. Identify header rows/columns if present
# 4. Handle merged cells (rowspan/colspan > 1)
# 5. Provide accurate cell coordinates
# 6. Assess extraction confidence

# Table information:
# - Source region ID: {table_region.region_id}
# - Original description: {table_region.content_description}

# The response must conform to the TableStructure schema with complete cell data.
# """
        
#         message = HumanMessage(
#             content=[
#                 {"type": "text", "text": prompt},
#                 {
#                     "type": "image_url", 
#                     "image_url": {"url": f"data:image/png;base64,{table_base64}"}
#                 }
#             ]
#         )
        
#         try:
#             # Get structured table result
#             table_result: TableStructure = self.table_analyzer.invoke([message])
            
#             # Update table ID and bbox to match source region
#             table_result.table_id = table_region.region_id
#             table_result.bbox = table_region.bbox
            
#             return table_result
            
#         except Exception as e:
#             logger.error(f"Table structure analysis failed: {str(e)}")
#             raise ValueError(f"Table analysis error: {str(e)}")
    
#     def extract_content_structured(self, image_path: str, 
#                                  text_regions: List[DocumentRegion]) -> ContentExtractionResult:
#         """Extract text content using structured output"""
        
#         base64_image = self._encode_image(image_path)
        
#         # Create region descriptions for prompt
#         region_descriptions = []
#         for region in text_regions:
#             if region.region_type in [RegionType.TEXT, RegionType.TITLE, RegionType.HEADER, RegionType.FOOTER]:
#                 desc = f"- {region.region_id} ({region.region_type.value}): {region.content_description}"
#                 region_descriptions.append(desc)
        
#         prompt = f"""
# Extract complete text content from all text-based regions in this document image.

# Text regions identified:
# {chr(10).join(region_descriptions)}

# Extraction requirements:
# 1. Extract complete, accurate text from each region
# 2. Preserve original formatting where possible
# 3. Detect primary language of the text
# 4. Assess text extraction quality for each region
# 5. Count words and characters accurately
# 6. Maintain text structure and line breaks

# Quality assessment criteria:
# - 1.0: Perfect, clear text extraction
# - 0.8-0.9: High quality with minor issues
# - 0.6-0.7: Good quality with some unclear characters
# - 0.4-0.5: Moderate quality with several issues
# - 0.0-0.3: Poor quality, significant extraction problems

# The response must conform to the ContentExtractionResult schema.
# """
        
#         message = HumanMessage(
#             content=[
#                 {"type": "text", "text": prompt},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
#                 }
#             ]
#         )
        
#         try:
#             # Get structured content extraction result
#             content_result: ContentExtractionResult = self.content_extractor.invoke([message])
#             return content_result
            
#         except Exception as e:
#             logger.error(f"Content extraction failed: {str(e)}")
#             raise ValueError(f"Content extraction error: {str(e)}")
