"""
Enhanced Triplet Extractor for JSON table data with structured output
File: src/analyzers/triplet_extractor.py
"""

from typing import List, Dict, Any, Optional, Union
from ..pydantic_models.knowledge_models import Entity, Relation, Triplet, EntityType, RelationType
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """Entity extraction result for structured output"""
    name: str = Field(description="Entity name or identifier")
    entity_type: EntityType = Field(description="Type of entity")
    value: Optional[str] = Field(default=None, description="Entity value if applicable")
    unit: Optional[str] = Field(default=None, description="Unit of measurement if applicable")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    description: Optional[str] = Field(default=None, description="Entity description")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    source_location: str = Field(description="Location in source data (row/column reference)")


class ExtractedRelation(BaseModel):
    """Relation extraction result for structured output"""
    subject_name: str = Field(description="Name of subject entity")
    predicate: RelationType = Field(description="Type of relation")
    object_name: str = Field(description="Name of object entity")
    confidence: float = Field(ge=0.0, le=1.0, description="Relation confidence")
    evidence: str = Field(description="Supporting evidence from source")
    context: Optional[str] = Field(default=None, description="Additional context")


class TripletExtractionOutput(BaseModel):
    """Structured output for triplet extraction"""
    document_type: str = Field(description="Type of document analyzed")
    extraction_summary: str = Field(description="Brief summary of extraction")
    
    entities: List[ExtractedEntity] = Field(
        description="All entities extracted from the data",
        min_items=0
    )
    
    relations: List[ExtractedRelation] = Field(
        description="All relations extracted from the data", 
        min_items=0
    )
    
    quality_assessment: Dict[str, float] = Field(
        description="Quality metrics for the extraction",
        default_factory=dict
    )
    
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Additional notes about the extraction process"
    )
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class TripletExtractor:
    """Enhanced knowledge triplet extractor using GPT-4o with structured output"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4096
        )
        
        # Create structured output analyzer
        self.triplet_analyzer = self.llm.with_structured_output(TripletExtractionOutput)
    
    def extract_triplets_from_table_data(self, table_data: Dict[str, Any], 
                                       page_number: int = 1,
                                       save_response: bool = False,
                                       output_dir: Optional[str] = None) -> List[Triplet]:
        """
        Extract triplets from structured table data (JSON format)
        
        Args:
            table_data: JSON-formatted table data from TableContentExtractor
            page_number: Page number for metadata
            save_response: Whether to save the raw LLM response
            output_dir: Directory to save response (if save_response=True)
            
        Returns:
            List of Triplet objects
        """
        logger.info(f"Extracting triplets from table data on page {page_number}")
        
        # Create extraction prompt for table data
        prompt = self._create_table_extraction_prompt(table_data)
        
        # Call GPT-4o with structured output
        message = HumanMessage(content=prompt)
        
        try:
            # Get structured result directly as Pydantic object
            extraction_result: TripletExtractionOutput = self.triplet_analyzer.invoke([message])
            
            logger.info(f"Successfully extracted {len(extraction_result.entities)} entities and {len(extraction_result.relations)} relations")
            
            # Save raw response if requested
            if save_response and output_dir:
                self._save_extraction_response(extraction_result, output_dir, page_number)
            
            # Convert to Triplet objects
            return self._convert_to_triplet_objects(extraction_result, table_data, page_number)
            
        except Exception as e:
            logger.error(f"Structured triplet extraction failed: {e}")
            return self._create_fallback_triplets(table_data, page_number)
    
    def extract_triplets_from_text(self, text: str, page_number: int = 1) -> List[Triplet]:
        """
        Legacy method: Extract triplets from plain text content
        """
        # Convert text to simple table-like structure
        text_data = {
            "structure": {"extraction_method": "text_input"},
            "data_rows": [{"content": text}],
            "metadata": {"source": "text_input"}
        }
        
        return self.extract_triplets_from_table_data(text_data, page_number)
    
    def _create_table_extraction_prompt(self, table_data: Dict[str, Any]) -> str:
        """Create extraction prompt for structured table data"""
        
        # Extract key information from table data
        structure_info = table_data.get("structure", {})
        data_rows = table_data.get("data_rows", [])
        metadata = table_data.get("metadata", {})
        
        # Format data rows for prompt - handle larger datasets
        data_preview = []
        max_rows_to_show = min(len(data_rows), 20)  # Increased from 5 to 20 rows
        
        for i, row in enumerate(data_rows[:max_rows_to_show]):
            data_preview.append(f"Row {i+1}: {json.dumps(row, ensure_ascii=False)}")
        
        if len(data_rows) > max_rows_to_show:
            data_preview.append(f"... and {len(data_rows) - max_rows_to_show} more rows")
            # Add sample of remaining rows to give LLM more context
            if len(data_rows) > max_rows_to_show + 5:
                data_preview.append("Sample from remaining rows:")
                sample_start = len(data_rows) - 3  # Last 3 rows as sample
                for i in range(sample_start, len(data_rows)):
                    row = data_rows[i]
                    data_preview.append(f"Row {i+1}: {json.dumps(row, ensure_ascii=False)}")
        
        logger.info(f"Sending {max_rows_to_show} rows (out of {len(data_rows)} total) to LLM for analysis")
        
        prompt = f"""
Analyze this structured table data and extract knowledge triplets with semantic understanding.

TABLE INFORMATION:
- Table ID: {structure_info.get('id', 'unknown')}
- Total Rows: {len(data_rows)} (showing up to {max_rows_to_show} rows)
- Dimensions: {structure_info.get('rows', '?')}x{structure_info.get('cols', '?')}
- Extraction Method: {structure_info.get('extraction_method', 'unknown')}
- Table Type: {metadata.get('table_type', 'unknown')}
- Language: {metadata.get('detected_language', 'unknown')}
- Document Type: {metadata.get('document_type', 'unknown')}

DATA CONTENT:
{chr(10).join(data_preview)}

IMPORTANT: You are seeing a sample of the data. Extract entities and relationships from ALL visible rows, and infer common patterns that might apply to the {len(data_rows)} total rows.

EXTRACTION REQUIREMENTS:

1. **ENTITY IDENTIFICATION** (use ONLY these types):
   - Parameters/variables (e.g., "durchfluss", "druck", "temperatur") → concept
   - Numerical values with units (e.g., "20 l/min", "5.2 bar") → numeric  
   - Pure numbers (e.g., "20", "50") → numeric
   - Locations/places (e.g., "Kesselhaus") → location
   - Equipment/products → product
   - Technical specifications → technology
   - Organizations/companies → organization
   - Other relevant entities → other

2. **RELATION IDENTIFICATION** (use ONLY these types):
   - Parameter has value → related_to
   - Parameter has unit → related_to
   - Min/Max relationships → related_to
   - Location relationships → located_in
   - Part-of relationships → part_of
   - Specification relationships → related_to
   - Measurement relationships → related_to
   - Technical relationships → related_to
   - Contains relationships → contains
   - Dependency relationships → depends_on

3. **CRITICAL CONSISTENCY REQUIREMENTS**:
   - EVERY entity mentioned in relations MUST also appear in the entities list
   - Use EXACT same names for entities in both entities and relations sections
   - If you create a relation "A -> B", both "A" and "B" MUST be listed as entities
   - Double-check entity name consistency before finalizing output

4. **VALID ENTITY TYPES**: person, organization, location, date, product, concept, numeric, event, technology, document, other

5. **VALID RELATION TYPES**: is_a, part_of, located_in, works_for, created_by, contains, related_to, temporal, causal, member_of, instance_of, similar_to, depends_on, used_by, owns, affects, precedes, follows, composed_of, produces, other

6. **QUALITY ASSESSMENT**:
   - Provide confidence scores for each extraction
   - Assess overall extraction quality
   - Note any ambiguities or uncertainties

FOCUS AREAS:
- Technical parameters and their values
- Measurement relationships with units
- Equipment specifications and properties
- Location and organizational information
- Range specifications (min/max values)

GERMAN TECHNICAL TERMS:
Handle German technical vocabulary appropriately:
- "durchfluss" (flow rate), "druck" (pressure), "temperatur" (temperature)
- "Stellort" (installation location), "MSR-Aufgabe" (control task)
- Technical units and measurement terms

EXAMPLE FOR CONSISTENCY:
If you extract relation: "durchfluss" related_to "20-50 l/min"
Then BOTH entities must be listed:
- Entity 1: name="durchfluss", entity_type="concept"
- Entity 2: name="20-50 l/min", entity_type="numeric"

CRITICAL: Only use the exact enum values listed above. Ensure perfect name matching between entities and relations.

The output must conform exactly to the TripletExtractionOutput schema.
Focus on creating meaningful knowledge relationships from the technical data.
"""
        
        return prompt
    
    def _convert_to_triplet_objects(self, extraction_result: TripletExtractionOutput,
                                  table_data: Dict[str, Any], 
                                  page_number: int) -> List[Triplet]:
        """Convert extraction result to Triplet objects"""
        
        # Create entity mapping
        entities_map = {}
        
        # Convert extracted entities to Entity objects
        for extracted_entity in extraction_result.entities:
            entity = Entity(
                name=extracted_entity.name,
                entity_type=extracted_entity.entity_type,
                aliases=extracted_entity.aliases,
                description=extracted_entity.description,
                confidence=extracted_entity.confidence,
                source_text=str(table_data.get("data_rows", []))[:200],
                page_number=page_number,
                metadata={
                    "value": extracted_entity.value,
                    "unit": extracted_entity.unit,
                    "source_location": extracted_entity.source_location,
                    "extraction_method": "structured_table_analysis"
                }
            )
            entities_map[extracted_entity.name] = entity
        
        # Convert relations to Triplet objects
        triplets = []
        missing_entities = set()
        
        for extracted_relation in extraction_result.relations:
            subject_name = extracted_relation.subject_name
            object_name = extracted_relation.object_name
            
            # Create missing entities if they don't exist
            if subject_name not in entities_map:
                missing_entities.add(subject_name)
                subject_entity = Entity(
                    name=subject_name,
                    entity_type=EntityType.OTHER,  # Default type for missing entities
                    confidence=0.7,
                    source_text=extracted_relation.evidence,
                    page_number=page_number,
                    metadata={
                        "auto_created": True,
                        "extraction_method": "relation_derived"
                    }
                )
                entities_map[subject_name] = subject_entity
                logger.info(f"Auto-created missing subject entity: {subject_name}")
            
            if object_name not in entities_map:
                missing_entities.add(object_name)
                object_entity = Entity(
                    name=object_name,
                    entity_type=EntityType.OTHER,  # Default type for missing entities
                    confidence=0.7,
                    source_text=extracted_relation.evidence,
                    page_number=page_number,
                    metadata={
                        "auto_created": True,
                        "extraction_method": "relation_derived"
                    }
                )
                entities_map[object_name] = object_entity
                logger.info(f"Auto-created missing object entity: {object_name}")
            
            # Now create the triplet (both entities should exist)
            triplet = Triplet(
                subject=entities_map[subject_name],
                predicate=extracted_relation.predicate,
                object=entities_map[object_name],
                confidence=extracted_relation.confidence,
                source_sentence=extracted_relation.evidence,
                page_number=page_number,
                extraction_method="structured_llm_analysis",
                metadata={
                    "context": extracted_relation.context,
                    "table_source": True,
                    "quality_metrics": extraction_result.quality_assessment,
                    "has_auto_created_entities": len(missing_entities) > 0
                }
            )
            triplets.append(triplet)
        
        if missing_entities:
            logger.warning(f"Auto-created {len(missing_entities)} missing entities: {list(missing_entities)[:5]}...")
        
        logger.info(f"Created {len(triplets)} triplets from {len(extraction_result.entities)} original entities + {len(missing_entities)} auto-created entities")
        return triplets
    
    def _create_fallback_triplets(self, table_data: Dict[str, Any], 
                                page_number: int) -> List[Triplet]:
        """Create simple fallback triplets when structured extraction fails"""
        logger.warning("Using fallback triplet extraction")
        
        fallback_triplets = []
        data_rows = table_data.get("data_rows", [])
        
        for i, row in enumerate(data_rows[:3]):  # Process first 3 rows
            row_header = row.get("row_header", f"parameter_{i}")
            
            # Create parameter entity
            param_entity = Entity(
                name=row_header,
                entity_type=EntityType.CONCEPT,
                confidence=0.6,
                source_text=str(row)[:100],
                page_number=page_number,
                metadata={"fallback_extraction": True}
            )
            
            # Create value entities and relations
            for key, value in row.items():
                if key != "row_header" and value:
                    value_entity = Entity(
                        name=str(value),
                        entity_type=EntityType.OTHER,
                        confidence=0.6,
                        source_text=str(row)[:100],
                        page_number=page_number,
                        metadata={"fallback_extraction": True}
                    )
                    
                    triplet = Triplet(
                        subject=param_entity,
                        predicate=RelationType.RELATED_TO,  # Fixed: use valid enum value
                        object=value_entity,
                        confidence=0.6,
                        source_sentence=f"{row_header} has {key} of {value}",
                        page_number=page_number,
                        extraction_method="fallback_rule_based",
                        metadata={"fallback_used": True}
                    )
                    fallback_triplets.append(triplet)
        
        return fallback_triplets
    
    def _save_extraction_response(self, extraction_result: TripletExtractionOutput,
                                output_dir: str, page_number: int):
        """Save the raw extraction response as JSON"""
        import os
        import json
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"triplet_extraction_page_{page_number}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        response_data = {
            "extraction_timestamp": timestamp,
            "page_number": page_number,
            "extraction_result": extraction_result.model_dump(),
            "model_info": {
                "model_name": "gpt-4o",
                "extraction_method": "structured_triplet_analysis",
                "version": "1.0"
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved triplet extraction response to: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save extraction response: {e}")
    
    def extract_from_multiple_tables(self, table_data_list: List[Dict[str, Any]],
                                   save_response: bool = False,
                                   output_dir: Optional[str] = None) -> List[Triplet]:
        """
        Extract triplets from multiple table data sources
        
        Args:
            table_data_list: List of table data dictionaries
            save_response: Whether to save responses
            output_dir: Output directory for responses
            
        Returns:
            Combined list of triplets from all tables
        """
        all_triplets = []
        
        for i, table_data in enumerate(table_data_list):
            try:
                page_number = table_data.get("metadata", {}).get("page_number", i + 1)
                
                triplets = self.extract_triplets_from_table_data(
                    table_data=table_data,
                    page_number=page_number,
                    save_response=save_response,
                    output_dir=output_dir
                )
                
                # Add table index to metadata
                for triplet in triplets:
                    triplet.metadata["table_index"] = i
                    triplet.metadata["source_table_id"] = table_data.get("structure", {}).get("id")
                
                all_triplets.extend(triplets)
                
            except Exception as e:
                logger.error(f"Failed to extract triplets from table {i}: {e}")
                continue
        
        logger.info(f"Extracted total of {len(all_triplets)} triplets from {len(table_data_list)} tables")
        return all_triplets


# Usage example
def example_usage():
    """Example of using the enhanced TripletExtractor with table data"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    
    # Initialize extractor
    triplet_extractor = TripletExtractor(api_key)
    
    # Example table data (from TableContentExtractor output)
    table_data = {
        "structure": {
            "id": "table_1",
            "rows": 5,
            "cols": 2,
            "extraction_method": "gpt4v_visual",
            "structure_confidence": 0.92
        },
        "data_rows": [
            {"row_header": "durchfluss", "value": "20-50 l/min"},
            {"row_header": "druck", "value": "2.5-8.0 bar"},
            {"row_header": "temperatur", "value": "15-80 °C"},
            {"row_header": "Stellort", "value": "Kesselhaus"},
            {"row_header": "MSR-Aufgabe", "value": "Durchflussmessung"}
        ],
        "metadata": {
            "table_type": "specification_table",
            "detected_language": "de",
            "document_type": "technical_datasheet",
            "page_number": 1,
            "main_topic": "Technical specifications"
        }
    }
    
    try:
        print("=== Triplet Extraction from Table Data ===")
        
        # Extract triplets with response saving
        triplets = triplet_extractor.extract_triplets_from_table_data(
            table_data=table_data,
            page_number=1,
            save_response=True,
            output_dir="data/outputs/triplet_responses"
        )
        
        print(f"Extracted {len(triplets)} triplets:")
        
        # Display results
        for i, triplet in enumerate(triplets, 1):
            print(f"\nTriplet {i}:")
            print(f"  Subject: {triplet.subject.name} ({triplet.subject.entity_type})")
            print(f"  Predicate: {triplet.predicate}")
            print(f"  Object: {triplet.object.name} ({triplet.object.entity_type})")
            print(f"  Confidence: {triplet.confidence:.2f}")
            print(f"  Source: {triplet.source_sentence[:100]}...")
        
        # Show entity statistics
        entities = set()
        for triplet in triplets:
            entities.add((triplet.subject.name, triplet.subject.entity_type))
            entities.add((triplet.object.name, triplet.object.entity_type))
        
        print(f"\n=== Entity Summary ===")
        print(f"Total unique entities: {len(entities)}")
        
        entity_types = {}
        for name, entity_type in entities:
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        for entity_type, count in entity_types.items():
            print(f"  {entity_type}: {count}")
        
        return triplets
        
    except Exception as e:
        print(f"Triplet extraction failed: {e}")
        return []


def example_usage_from_json_file():
    """Example of processing TripletExtractor using saved TableContentExtractor JSON files"""
    import os
    import json
    import glob
    from dotenv import load_dotenv
    load_dotenv()

    # Configuration
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    json_input_dir = "data/outputs/temp_images/llm_responses"  # Directory with saved JSON files
    triplet_output_dir = "data/outputs/triplet_responses"  # Output directory for triplet responses
    
    # Initialize extractor
    triplet_extractor = TripletExtractor(api_key)
    
    print("=== Triplet Extraction from Saved JSON Files ===")
    print(f"Input directory: {json_input_dir}")
    print(f"Output directory: {triplet_output_dir}")
    
    try:
        # Find all LLM response JSON files
        json_pattern = os.path.join(json_input_dir, "llm_response_page_*.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            print(f"No JSON files found in {json_input_dir}")
            print("Make sure you have run TableContentExtractor with save_response=True first")
            return []
        
        print(f"Found {len(json_files)} JSON file(s) to process")
        
        all_triplets = []
        
        for json_file in sorted(json_files):
            print(f"\n--- Processing: {os.path.basename(json_file)} ---")
            
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Extract page number and raw semantic result
            page_number = json_data.get("page_number", 1)
            raw_semantic_result = json_data.get("raw_semantic_result", {})
            
            print(f"Page number: {page_number}")
            print(f"Extraction timestamp: {json_data.get('extraction_timestamp', 'unknown')}")
            
            # Process each table in the raw semantic result
            tables = raw_semantic_result.get("tables", [])
            print(f"Found {len(tables)} table(s) in semantic result")
            
            for table_idx, table in enumerate(tables):
                print(f"\n  Processing Table {table_idx + 1}: {table.get('title', 'Untitled')}")
                print(f"  Table type: {table.get('table_type', 'unknown')}")
                print(f"  Data relationships: {len(table.get('data_relationships', []))}")
                
                # Convert semantic table data to the format expected by TripletExtractor
                converted_table_data = convert_semantic_table_to_triplet_input(
                    table, raw_semantic_result, page_number
                )
                
                # Extract triplets from this table
                try:
                    triplets = triplet_extractor.extract_triplets_from_table_data(
                        table_data=converted_table_data,
                        page_number=page_number,
                        save_response=True,
                        output_dir=triplet_output_dir
                    )
                    
                    print(f"  Extracted {len(triplets)} triplets from this table")
                    
                    # Add source file info to triplet metadata
                    for triplet in triplets:
                        triplet.metadata.update({
                            "source_json_file": os.path.basename(json_file),
                            "table_index": table_idx,
                            "source_table_title": table.get('title')
                        })
                    
                    all_triplets.extend(triplets)
                    
                except Exception as e:
                    print(f"  Failed to extract triplets from table {table_idx + 1}: {e}")
                    continue
        
        # Display overall results
        print(f"\n=== Overall Extraction Results ===")
        print(f"Total files processed: {len(json_files)}")
        print(f"Total triplets extracted: {len(all_triplets)}")
        
        if all_triplets:
            # Show sample triplets
            print(f"\n=== Sample Triplets ===")
            for i, triplet in enumerate(all_triplets[:5], 1):  # Show first 5 triplets
                print(f"\nTriplet {i}:")
                print(f"  Subject: {triplet.subject.name} ({triplet.subject.entity_type})")
                print(f"  Predicate: {triplet.predicate}")
                print(f"  Object: {triplet.object.name} ({triplet.object.entity_type})")
                print(f"  Confidence: {triplet.confidence:.2f}")
                print(f"  Source table: {triplet.metadata.get('source_table_title', 'unknown')}")
                print(f"  Source file: {triplet.metadata.get('source_json_file', 'unknown')}")
            
            if len(all_triplets) > 5:
                print(f"\n  ... and {len(all_triplets) - 5} more triplets")
            
            # Entity type statistics
            entity_type_stats = {}
            relation_type_stats = {}
            
            for triplet in all_triplets:
                # Count entity types
                subj_type = triplet.subject.entity_type
                obj_type = triplet.object.entity_type
                entity_type_stats[subj_type] = entity_type_stats.get(subj_type, 0) + 1
                entity_type_stats[obj_type] = entity_type_stats.get(obj_type, 0) + 1
                
                # Count relation types
                rel_type = triplet.predicate
                relation_type_stats[rel_type] = relation_type_stats.get(rel_type, 0) + 1
            
            print(f"\n=== Entity Type Distribution ===")
            for entity_type, count in sorted(entity_type_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {entity_type}: {count}")
            
            print(f"\n=== Relation Type Distribution ===")
            for relation_type, count in sorted(relation_type_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {relation_type}: {count}")
        
        return all_triplets
        
    except Exception as e:
        print(f"JSON processing failed: {e}")
        return []


def convert_semantic_table_to_triplet_input(table: Dict[str, Any], 
                                          semantic_result: Dict[str, Any],
                                          page_number: int) -> Dict[str, Any]:
    """
    Convert semantic table data from TableContentExtractor JSON to TripletExtractor input format
    
    Args:
        table: Single table from raw_semantic_result.tables
        semantic_result: The complete raw_semantic_result
        page_number: Page number from the JSON file
        
    Returns:
        Dictionary in the format expected by TripletExtractor
    """
    
    # Extract data relationships and convert to data_rows format
    data_relationships = table.get("data_relationships", [])
    data_rows = []
    
    logger.debug(f"Converting table with {len(data_relationships)} data relationships")
    
    for i, relationship in enumerate(data_relationships):
        row_header = relationship.get("row_header", "")
        values = relationship.get("values", {})
        single_value = relationship.get("single_value")
        row_notes = relationship.get("row_notes")
        
        # Create a row entry
        row_entry = {"row_header": row_header}
        
        # Add values
        if single_value and not values:
            # Simple parameter -> value format
            row_entry["value"] = single_value
            logger.debug(f"Row {i+1}: {row_header} -> {single_value}")
        else:
            # Multi-column format
            row_entry.update(values)
            logger.debug(f"Row {i+1}: {row_header} -> {values}")
        
        # Add notes if present
        if row_notes:
            row_entry["notes"] = row_notes
        
        data_rows.append(row_entry)
    
    # Convert to TripletExtractor input format
    converted_data = {
        "structure": {
            "id": table.get("table_id", "unknown"),
            "rows": table.get("structure", {}).get("rows", len(data_rows)),
            "cols": table.get("structure", {}).get("columns", 2),
            "extraction_method": "semantic_table_extraction",
            "structure_confidence": semantic_result.get("extraction_metadata", {}).get("confidence", 0.8)
        },
        "data_rows": data_rows,
        "metadata": {
            "table_type": table.get("table_type", "unknown"),
            "detected_language": semantic_result.get("extraction_metadata", {}).get("detected_language", "unknown"),
            "document_type": semantic_result.get("table_summary", {}).get("document_type", "unknown"),
            "page_number": page_number,
            "main_topic": semantic_result.get("table_summary", {}).get("main_topic", ""),
            "table_title": table.get("title"),
            "table_notes": table.get("notes"),
            "complexity_level": semantic_result.get("extraction_metadata", {}).get("complexity_level", "unknown"),
            "data_types": semantic_result.get("extraction_metadata", {}).get("data_types", []),
            "total_tables": semantic_result.get("table_summary", {}).get("total_tables", 1)
        }
    }
    
    logger.info(f"Converted table '{table.get('title', 'Untitled')}' with {len(data_rows)} rows to triplet input format")
    return converted_data


if __name__ == "__main__":
    # Run the original example
    print("Running original example with mock data...")
    results1 = example_usage()
    
    print("\n" + "="*50 + "\n")
    
    # Run the JSON file processing example
    print("Running JSON file processing example...")
    results2 = example_usage_from_json_file()