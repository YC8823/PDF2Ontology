from typing import List, Dict, Any, Optional
from ..pydantic_models.knowledge_models import Entity, Relation, Triplet, EntityType, RelationType
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json

class TripletExtractionResult(BaseModel):
    """Triplet extraction result model"""
    entities: List[Dict[str, Any]] = Field(description="Entity list")
    relations: List[Dict[str, Any]] = Field(description="Relation list")
    triplets: List[Dict[str, Any]] = Field(description="Triplet list")

class TripletExtractor:
    """Knowledge triplet extractor using GPT-4o"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4000
        )
        self.parser = PydanticOutputParser(pydantic_object=TripletExtractionResult)
    
    def extract_triplets_from_text(self, text: str, page_number: int = 1) -> List[Triplet]:
        """Extract triplets from text content"""
        prompt = self._create_extraction_prompt(text)
        
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        
        try:
            result = self.parser.parse(response.content)
            return self._convert_to_triplets(result, text, page_number)
        except Exception as e:
            print(f"Triplet extraction parsing error: {e}")
            return self._extract_with_fallback(text, page_number)
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create triplet extraction prompt"""
        return f"""
Please extract knowledge triplets from the following text. Identify entities, relationships, and construct triplets.

Text content:
{text}

Please extract according to the following requirements:

1. **Entity Recognition**: Identify key entities in the text, including:
   - People (person)
   - Organizations (organization)  
   - Locations (location)
   - Dates/times (date)
   - Products (product)
   - Concepts (concept)
   - Numbers/values (numeric)
   - Other (other)

2. **Relationship Recognition**: Identify relationships between entities, including:
   - Is-a relationship (is_a)
   - Part-of relationship (part_of)
   - Location relationship (located_in)
   - Work relationship (works_for)
   - Creation relationship (created_by)
   - Containment relationship (contains)
   - Related relationship (related_to)
   - Temporal relationship (temporal)
   - Causal relationship (causal)
   - Other relationships (other)

3. **Triplet Construction**: Combine identified entities and relationships into triplets (subject, predicate, object)

{self.parser.get_format_instructions()}

Please ensure:
- Standardized entity names
- Accurate relationship types
- Logical triplets
- Include confidence assessment
"""
    
    def _convert_to_triplets(self, result: TripletExtractionResult, 
                           source_text: str, page_number: int) -> List[Triplet]:
        """Convert extraction result to triplet objects"""
        # Create entity mapping
        entities_map = {}
        entities = []
        
        for entity_data in result.entities:
            entity = Entity(
                name=entity_data["name"],
                entity_type=EntityType(entity_data.get("type", "other")),
                aliases=entity_data.get("aliases", []),
                description=entity_data.get("description"),
                confidence=entity_data.get("confidence", 0.8),
                source_text=source_text[:200] + "..." if len(source_text) > 200 else source_text,
                page_number=page_number
            )
            entities.append(entity)
            entities_map[entity.name] = entity
        
        # Create triplets
        triplets = []
        for triplet_data in result.triplets:
            subject_name = triplet_data["subject"]
            object_name = triplet_data["object"]
            
            if subject_name in entities_map and object_name in entities_map:
                triplet = Triplet(
                    subject=entities_map[subject_name],
                    predicate=RelationType(triplet_data.get("predicate", "related_to")),
                    object=entities_map[object_name],
                    confidence=triplet_data.get("confidence", 0.8),
                    source_sentence=triplet_data.get("source_sentence", source_text),
                    page_number=page_number
                )
                triplets.append(triplet)
        
        return triplets
    
    def _extract_with_fallback(self, text: str, page_number: int) -> List[Triplet]:
        """Fallback triplet extraction method using simple rules"""
        import re
        
        triplets = []
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            # Simple pattern matching
            # More complex rules can be added here
            words = sentence.split()
            if len(words) >= 3:
                # Create simple triplets
                subject = Entity(
                    name=words[0],
                    entity_type=EntityType.OTHER,
                    confidence=0.6,
                    source_text=sentence,
                    page_number=page_number
                )
                
                obj = Entity(
                    name=words[-1],
                    entity_type=EntityType.OTHER,
                    confidence=0.6,
                    source_text=sentence,
                    page_number=page_number
                )
                
                triplet = Triplet(
                    subject=subject,
                    predicate=RelationType.RELATED_TO,
                    object=obj,
                    confidence=0.6,
                    source_sentence=sentence,
                    page_number=page_number
                )
                triplets.append(triplet)
        
        return triplets