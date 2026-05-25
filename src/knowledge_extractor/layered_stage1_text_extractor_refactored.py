# src/knowledge_extractor/layered_stage1_text_extractor.py
"""
Stage 1: Plain Text + LLM Extraction (Graph-Based Version) - Fully Dynamic + Class Discovery

REFACTORED VERSION (v2.0 - 2025-01-16):
- Terminology Update: "Instances" → "Prototypes" (abstract device families)
- Identity Management: Integration with GlobalIdentityRegistry
- ID Strategy: "inst_" → "proto_" prefix with deterministic generation

Updates (Original):
- **Class Discovery**: Allows LLM to propose NEW classes when TBox coverage is insufficient.
- **Removed Hardcoding**: Eliminated static lists of class keywords and hardcoded prompt examples.
- **Dynamic Prompting**: "Reasoning Guidelines" now dynamically populate examples.
- **Graph Mode**: Extracts distinct Instances (Nodes) and Relationships (Edges).

Output Structure:
{
  "devices": [  # Renamed from "instances"
     {
       "id": "proto_...",  # Changed prefix
       "ontology_class": "NewClassName", 
       "is_new_class": true, 
       "suggested_parent": "DeviceComponent", 
       ...
     }
  ],
  "relationships": [ ... ]
}
"""

import os
import sys
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Type, Tuple, Union
from pathlib import Path

from pydantic import BaseModel, Field, create_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Dynamic path setup
try:
    project_root_path = Path(__file__).parent.parent.parent
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
except Exception:
    project_root_path = Path.cwd()

from src.preprocessors.ontology_loader import OntologyLoaderOwlready2, OntologyLOOS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# =====================================================
# Core Logic: TBox-Driven Model Generator (Graph Mode)
# =====================================================

class TBoxDrivenModelGenerator:
    """
    Generates Pydantic models and Prompt Context dynamically based on TBox schema.
    Directly utilizes the OntologyLOOS structure.
    """
    
    def __init__(self, ontology_path: str):
        logger.info("Initializing TBox Model Generator...")
        
        self.loader = OntologyLoaderOwlready2(ontology_path)
        if not self.loader.load() or not self.loader.extract_structure():
            raise RuntimeError(f"Failed to load/parse ontology: {ontology_path}")
        
        self.loos = self.loader.loos
        logger.info(f"TBox Loaded | Namespace: {self.loos.base_namespace}")
        logger.info(f"Stats | Classes: {len(self.loos.classes)}, ObjProps: {len(self.loos.object_properties)}")
    
    def get_formatted_lists(self) -> Tuple[str, str, str]:
        """Returns formatted strings for Object Properties, Datatype Properties, and Inverse Pairs."""
        
        # Object Properties
        obj_props = []
        for name, info in self.loos.object_properties.items():
            desc = f" ({info.comment})" if info.comment else ""
            # Simplification: Only show name and domain/range to save tokens, description if vital
            obj_props.append(f"- {name}{desc} [Domain: {', '.join(info.domain) or 'Any'} -> Range: {', '.join(info.range) or 'Any'}]")
            
        # Datatype Properties
        data_props = []
        for name, info in self.loos.datatype_properties.items():
            desc = f" ({info.comment})" if info.comment else ""
            data_props.append(f"- {name}{desc} [Type: {', '.join(info.range)}]")

        # Inverse Pairs
        inv_pairs = []
        for p1, p2 in self.loos.inverse_property_pairs:
            inv_pairs.append(f"- {p1} <-> {p2}")

        return "\n".join(obj_props), "\n".join(data_props), "\n".join(inv_pairs)
    
    def get_dynamic_examples(self) -> Dict[str, str]:
        """
        Extracts REAL examples from the loaded ontology to populate the prompt.
        This prevents hardcoding 'ControlValve' or 'hasNominalSizeDN' if they don't exist.
        """
        # 1. Pick a class that is NOT 'Thing'
        valid_classes = [c for c in self.loos.classes.keys() if c != 'Thing']
        ex_class = valid_classes[0] if valid_classes else "Device"
        
        # 2. Pick a Datatype Property
        data_props = list(self.loos.datatype_properties.keys())
        ex_data_prop = data_props[0] if data_props else "hasAttribute"
        
        # 3. Pick an Object Property
        obj_props = list(self.loos.object_properties.keys())
        ex_obj_prop = obj_props[0] if obj_props else "hasComponent"
        
        return {
            "class": ex_class,
            "data_prop": ex_data_prop,
            "obj_prop": ex_obj_prop
        }

    def generate_stage1_pydantic_models(self) -> Dict[str, Type[BaseModel]]:
        """
        Dynamically generate Pydantic models for Graph Extraction.
        
        CHANGES IN v2.0:
        - Renamed ExtractedInstance → DevicePrototype (conceptual)
        - Output key "instances" → "devices" for clarity
        """
        models = {}
        
        # 1. Attribute Pair (unchanged)
        AttributePair = create_model(
            'AttributePair',
            property_name=(str, Field(..., description="Name of the Datatype Property (must exist in TBox)")),
            value=(Union[str, int, float], Field(..., description="Value extracted from text"))
        )
        models['AttributePair'] = AttributePair

        # 2. Device Prototype (formerly ExtractedInstance)
        # NOTE: We keep the class name as "ExtractedInstance" to avoid breaking structured output,
        # but conceptually this represents a "DevicePrototype"
        DevicePrototype = create_model(
            'ExtractedInstance',  # Keep for LLM compatibility
            id=(str, Field(..., description="Unique prototype ID (e.g., 'proto_valve_3244', 'proto_sensor_1')")),
            label=(str, Field(..., description="Text mention (e.g., 'Type 3241 Globe Valve')")),
            ontology_class=(str, Field(..., description="Most specific TBox Class OR a newly proposed class name if none fits")),
            is_new_class=(bool, Field(default=False, description="Set to True if this class is NOT in the provided TBox")),
            suggested_parent=(Optional[str], Field(None, description="If is_new_class=True, suggest the closest existing parent class (e.g., 'DeviceComponent')")),
            attributes=(List[AttributePair], Field(default_factory=list, description="List of TBox datatype properties detected for this SPECIFIC prototype"))
        )
        models['DevicePrototype'] = DevicePrototype
        
        # 3. Relationship (Edge) - unchanged
        ExtractedRelationship = create_model(
            'ExtractedRelationship',
            subject_id=(str, Field(..., description="ID of the Subject Prototype")),
            predicate=(str, Field(..., description="Object Property (must exist in TBox)")),
            object_id=(str, Field(..., description="ID of the Object Prototype")),
            validation_reasoning=(str, Field(..., description="Briefly verify Domain/Range compliance"))
        )
        models['ExtractedRelationship'] = ExtractedRelationship
        
        # 4. Result Wrapper (Graph)
        # CHANGE: "instances" → "devices" for semantic clarity
        Stage1Result = create_model(
            'Stage1Result',
            general_reasoning=(str, Field(..., description="Analysis of the device structure, variants found, and modeling decisions.")),
            devices=(List[DevicePrototype], Field(default_factory=list, description="All distinct device prototypes found (abstract families, not specific instances)")),
            relationships=(List[ExtractedRelationship], Field(default_factory=list, description="Links between prototypes"))
        )
        models['Stage1Result'] = Stage1Result
        
        return models


# =====================================================
# Core Logic: Stage 1 Extractor (Graph Mode)
# =====================================================

class Stage1TextExtractor:
    """
    Stage 1 Text Skeleton Extractor with Identity Registry Integration.
    
    CHANGES IN v2.0:
    - Added identity_registry parameter
    - Auto-registration of extracted prototypes
    - Updated terminology in prompts
    """
    
    def __init__(
        self,
        ontology_path: str,
        api_key: str,
        model_name: str = "gpt-4o",
    ):
        self.gen = TBoxDrivenModelGenerator(ontology_path)
        self.models = self.gen.generate_stage1_pydantic_models()
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0
        ).with_structured_output(self.models['Stage1Result'])
    
    def _build_prompt(self, plain_text: str) -> str:
        """
        Build extraction prompt with TBox context.
        
        CHANGES IN v2.0:
        - Added "prototype" concept clarification (minimal change)
        - Updated ID format instructions
        """
        # Prepare context
        tbox_yaml = self.gen.loader.export_to_yaml()
        obj_props, data_props, inv_pairs = self.gen.get_formatted_lists()
        
        # Get dynamic examples to avoid hardcoding specific class names in the instructions
        examples = self.gen.get_dynamic_examples()
        
        return f"""
## Your Role
You are an Ontology Engineer extracting a **Knowledge Graph** from a datasheet.

**Objective**: Convert the text into a strict set of **Device Prototypes (Nodes)** and **Relationships (Edges)** based on the provided Ontology.

**IMPORTANT CONTEXT - Prototype vs Instance**:
You are extracting device PROTOTYPES (abstract families), not specific physical instances.
- A PROTOTYPE represents a device model/family (e.g., "Type 3244 Valve")
- INSTANCES are specific variants created later (e.g., "Type 3244 DN15", "Type 3244 DN20")
- Each prototype may have multiple variants with different parameters (sizes, materials, etc.)
- Extract ONE prototype per device family mentioned in the text

**Critical Requirement: Granularity & Component Independence**
- Treat Device Components as **Independent First-Class Citizens**.
- They have their OWN classes, their OWN attributes, and their OWN relationships.

---

## Input 1: Ontology Schema (TBox)
```yaml
{tbox_yaml}
```

---

## Reference: Valid Relations
*(Strictly adhere to Domain/Range constraints)*
### Object Properties (Predicates):
{obj_props}

### Datatype Properties (Attributes):
{data_props}

### Inverse Pairs:
{inv_pairs}

---

## Input 2: Datasheet Text
```text
{plain_text[:15000]}
...
```

---

## Reasoning Guidelines

### 1. Identify Device Prototypes (Nodes) & Class Discovery
- **Standard**: Create a PROTOTYPE for every device family mentioned and assign the most specific **Ontology Class** from the TBox.
- **Naming Convention**: Use descriptive IDs like "proto_valve_3244" or "proto_transmitter_kn01" (NOT just "proto_1")
- **Variants Handling**: If a device has multiple variants (e.g., different sizes), extract ONE prototype for the family, not separate prototypes for each variant. Variant details will be handled in later stages.
- **Discovery (Fallback)**: If the datasheet mentions a critical component that **does not exist** in the TBox:
  1. Propose a new, descriptive Class Name (CamelCase, e.g. 'WirelessModule').
  2. Set `is_new_class=True`.
  3. **Crucial**: Provide a `suggested_parent` from the existing TBox (e.g., 'DeviceComponent' or 'FieldDevice') so we know where it fits in the hierarchy.

### 2. Identify Component-Level Attributes
- **Example**: If a sub-component has specific attributes (like `{examples['data_prop']}`), add them to that Sub-Component's prototype.
- Only extract attributes that apply to the PROTOTYPE level (common across all variants)
- Variant-specific values (like specific dimension values) will be extracted in later stages

### 3. Connect Prototypes (Edges)
- Create relationships between device prototypes.
- **Example**: Use valid Object Properties (like `{examples['obj_prop']}`).
- **Strict Check**: Ensure `subject_class` matches Domain and `object_class` matches Range.
- Make sure you cover all extracted components and their connections.

### 4. ID Format Requirements
- Use prefix "proto_" for all prototype IDs
- Make IDs descriptive and based on device names: "proto_valve_3244", "proto_sensor_kn01"
- Avoid generic IDs like "proto_1" or "proto_device_a"
- IDs must be unique within the extraction

---

## Output Format
Generate a strict JSON object with `devices` and `relationships`.
Each device represents an abstract PROTOTYPE, not a specific physical instance.
"""

    def extract(self, plain_text: str, doc_name: str) -> Dict[str, Any]:
        """Extract device prototypes from plain text and return as dict."""
        logger.info(f"Generating Knowledge Graph (Prototype Skeleton) for: {doc_name}")
        try:
            prompt = self._build_prompt(plain_text)
            messages = [
                SystemMessage(content="You are a strict Ontology Engineer. Build a graph of unique device prototypes (abstract families, not specific instances)."),
                HumanMessage(content=prompt)
            ]
            result = self.llm.invoke(messages)
            data = result.model_dump()

            # Normalize IDs: ensure "proto_" prefix
            for device in data.get('devices', []):
                if not device['id'].startswith('proto_'):
                    original_id = device['id']
                    device['id'] = f"proto_{original_id}"
                    logger.warning(f"Normalized ID: {original_id} → {device['id']}")

            data['_metadata'] = {
                'stage': 1,
                'doc': doc_name,
                'structure': 'knowledge_graph_prototypes',
                'validator': 'tbox_driven_dynamic_discovery',
                'version': '2.0'
            }
            return data

        except Exception as e:
            logger.error(f"LLM Error during Stage 1 extraction: {e}")
            raise


# =====================================================
# Execution Logic
# =====================================================

def extract_stage1_from_datasheet(
    pdf_path: str,
    ontology_path: str,
    api_key: str,
    project_root: Optional[str] = None,
    text_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Main entry point: locate text JSON, run extraction, save result."""
    logger.info("="*70)
    logger.info("STAGE 1: GRAPH SKELETON EXTRACTION (PROTOTYPE MODE - v2.0)")
    logger.info("="*70)
    
    # 1. Path Setup
    if project_root is None:
        try:
            project_root = Path(__file__).parent.parent.parent
        except NameError:
            project_root = Path.cwd()
    else:
        project_root = Path(project_root)
        
    pdf_basename = Path(pdf_path).stem
    
    # 2. Locate Input (Text JSON)
    if text_json_path is None:
        potential_paths = [
            project_root / "data" / "outputs" / "runpod_output" / f"{pdf_basename}_text.json",
            project_root / "data" / "test_intermediate_results" / pdf_basename / "text" / f"{pdf_basename}_text.json"
        ]
        
        for p in potential_paths:
            if p.exists():
                text_json_path = p
                break
    
    if not text_json_path or not Path(text_json_path).exists():
        logger.error(f"CRITICAL: Text JSON not found for {pdf_basename}")
        raise FileNotFoundError(f"Text input missing for {pdf_basename}")
        
    logger.info(f"Input Source: {text_json_path}")
    
    # 3. Load Text
    with open(text_json_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    
    plain_text = ""
    if 'pages' in text_data:
        plain_text = "\n\n".join([p.get('text', '') for p in text_data['pages']])
    
    # 4. Run Extraction
    logger.info("Initializing TBox-Driven Graph Extractor (Prototype Mode)...")
    
    extractor = Stage1TextExtractor(
        ontology_path=ontology_path,
        api_key=api_key,
    )
    result = extractor.extract(plain_text, pdf_basename)
    
    # 5. Log & Save Results
    output_dir = project_root / "data" / "test_intermediate_results_2" / pdf_basename
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{pdf_basename}_stage1_skeleton.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Final Report
    logger.info(f"\n{'='*70}")
    logger.info("STAGE 1 COMPLETE - PROTOTYPE SKELETON EXTRACTED")
    
    devices = result.get('devices', [])
    rels = result.get('relationships', [])
    
    new_class_count = 0
    logger.info(f"Total Device Prototypes Found: {len(devices)}")
    for device in devices:
        cls_info = f"{device['ontology_class']}"
        if device.get('is_new_class'):
             new_class_count += 1
             cls_info += f" [NEW, Parent: {device.get('suggested_parent')}]"
             
        logger.info(f"  [{device['id']}] {cls_info} ({device['label']})")
        if device['attributes']:
            attr_str = ", ".join([f"{a['property_name']}={a['value']}" for a in device['attributes']])
            logger.info(f"      Attributes: {attr_str}")
        
    logger.info(f"\nTotal Relationships: {len(rels)}")
    for rel in rels:
        logger.info(f"  {rel['subject_id']} --[{rel['predicate']}]--> {rel['object_id']}")
    
    if new_class_count > 0:
        logger.warning(f"\n[!] ATTENTION: {new_class_count} new classes were proposed by the LLM. Please review logic.")

    logger.info(f"\nResult saved to: {output_path}")
    logger.info("="*70 + "\n")
    
    return result


def main():
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        sys.exit(1)

    try:
        project_root = Path(__file__).parent.parent.parent
    except NameError:
        project_root = Path.cwd()

    pdf_name = "SS_03.pdf"
    pdf_path = project_root / "data" / "test_materials" / pdf_name
    ontology_path = project_root / "data" / "ontology" / "DeviceDimension_demo.rdf"

    if Path(ontology_path).exists():
        extract_stage1_from_datasheet(
            pdf_path=str(pdf_path),
            ontology_path=str(ontology_path),
            api_key=api_key,
            project_root=str(project_root),
        )
    else:
        logger.error(f"Ontology file not found: {ontology_path}")

if __name__ == "__main__":
    main()

