# src/knowledge_extractor/layered_stage2_visual_extractor.py

"""
Stage 2: Visual Extraction & Graph Patching (Image + LLM) - Context-Aware & ID-Managed

REFACTORED VERSION (v2.0 - 2025-01-16):
- Terminology Update: "Instances" → "Prototypes" (enriched device families)
- GlobalIdentityRegistry Integration: Cross-stage ID resolution
- Enhanced Discovery: Support for new device prototypes and variants
- Dimension Prototype IDs: Deterministic generation based on device + parameter

Updates (Original):
- **Flag + Callback Architecture**: LLM suggests 'intent' with temporary IDs; Python middleware manages global UUIDs.
- **Context-Based Input**: Uses context_before/context_after.
- **TBox Aware**: Dynamically loads 'Dimension' class properties.

Input:
- stage1_output.json
- images.json
- ontology_file
- identity_registry (GlobalIdentityRegistry)

Output:
- {basename}_stage2_enriched_graph.json (renamed from merged_graph)
"""

import os
import sys
import json
import logging
import base64
import uuid
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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
# Pydantic Models (LLM Interface)
# =====================================================

class AttributePair(BaseModel):
    property_name: str = Field(..., description="TBox Property Name")
    value: Union[str, float, int] = Field(..., description="The extracted value.")

class NewDevicePrototype(BaseModel):
    """
    Intent to create or enrich a device prototype.
    LLM provides a SEMANTIC TEMP ID. Python Middleware will resolve it via GlobalIdentityRegistry.
    
    IMPORTANT CHANGE IN v2.0:
    - This represents a PROTOTYPE (abstract device family), not a specific instance
    - May be a NEW prototype OR enrichment of Stage 1 prototype
    - Variants (e.g., "Type 3244 with Feature X") should still be ONE prototype
    """
    temp_id: str = Field(..., description="A semantic temporary ID (e.g., 'temp_valve_3244', 'temp_valve_3244_variant')")
    ontology_class: str = Field(..., description="Inferred class, e.g., 'ValveBody', 'ControlValve'")
    label: str = Field(..., description="Exact device name from context (e.g., 'Type 3244', 'Type 3244 Variant A'). CRITICAL for deduplication.")
    attributes: List[AttributePair] = Field(default_factory=list, description="Inferred attributes for this prototype")
    inference_reasoning: str = Field(..., description="Why do you think this is a new/variant prototype? What context clues led to this?")
    is_variant: bool = Field(default=False, description="Set to True if this is a variant of an existing device family (e.g., 'Type 3244 with extra flange')")

class DimensionPrototype(BaseModel):
    """
    A dimension prototype attached to a device prototype.
    Will be fissioned into instances in Stage 3.

    IMPORTANT: This is a PROTOTYPE describing a dimension parameter,
    not an instance with specific numeric values yet.
    """
    temp_id: str = Field(..., description="Temp ID for the dimension prototype (e.g., 'dim_h1_3244')")
    parameter_name: str = Field(..., description="Dimension parameter name (e.g., 'H1', 'L', 'd')")
    dimension_type: str = Field(..., description="Must be exactly one of: 'vertical', 'horizontal', 'diameter', 'other'")
    label: str = Field(default="", description="Optional human-readable label (e.g., 'Height H1')")

class DimensionRelationship(BaseModel):
    subject_temp_id: str = Field(..., description="The TEMP ID of the Device Prototype (or existing Stage 1 ID if known)")
    predicate: str = Field("hasDimension", description="TBox Object Property")
    object_temp_id: str = Field(..., description="The TEMP ID of the Dimension Prototype")
    validation_reasoning: str = Field(..., description="Reasoning for this link")

class VisualGraphPatch(BaseModel):
    is_dimension_drawing: bool = Field(..., description="Is this a dimension drawing?")
    skip_reason: Optional[str] = Field(None, description="Reason if skipped")
    context_analysis: str = Field(..., description="Analysis of text surrounding the image")
    
    # LLM proposes device prototypes here (may be new or enrichment of existing)
    new_device_prototypes: List[NewDevicePrototype] = Field(
        default_factory=list, 
        description="Device PROTOTYPES found or enriched in this drawing"
    )
    dimension_prototypes: List[DimensionPrototype] = Field(
        default_factory=list, 
        description="Dimension PROTOTYPES extracted from drawing"
    )
    relationships: List[DimensionRelationship] = Field(
        default_factory=list, 
        description="Links between device prototypes and dimension prototypes"
    )

# =====================================================
# Core Logic: Stage 2 Extractor
# =====================================================

class Stage2VisualExtractor:
    """
    Stage 2 Visual Extractor with GlobalIdentityRegistry Integration.
    
    CHANGES IN v2.0:
    - Replaces internal GlobalIDRegistry with shared GlobalIdentityRegistry
    - Device/Dimension → DevicePrototype/DimensionPrototype terminology
    - Enhanced variant detection support
    - Deterministic dimension prototype ID generation
    """
    
    def __init__(
        self,
        api_key: str,
        ontology_path: str,
        model_name: str = "gpt-4o",
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=10000
        ).with_structured_output(VisualGraphPatch)

        self.ontology_path = ontology_path
        self.dimension_class_def = ""
        self.existing_prototypes_summary = ""
        # label → proto_id map built from Stage 1 output; updated when new devices are found
        self._label_to_id: Dict[str, str] = {}
        self.loos: Optional[OntologyLOOS] = None

        self._load_dimension_tbox()
        logger.info("✓ Stage 2 Visual Extractor initialized")

    def _load_dimension_tbox(self):
        """Loads Ontology and specifically 'Dimension' class info."""
        loader = OntologyLoaderOwlready2(self.ontology_path)
        if loader.load() and loader.extract_structure():
            self.loos = loader.loos
            dim_cls = self.loos.classes.get('Dimension')
            
            # Construct TBox Context
            props_desc = []

            for prop_name, prop_info in self.loos.datatype_properties.items():
                if 'Dimension' in prop_info.domain or not prop_info.domain:
                    rng = prop_info.range if prop_info.range else ["Any"]
                    props_desc.append(f"- {prop_name} (Type: {rng})")
            
            self.dimension_class_def = (
                f"### Class 'Dimension' Schema\n"
                f"Description: {dim_cls.comment if dim_cls else 'Physical measurements'}\n"
                f"**Allowed Attributes:**\n" + "\n".join(props_desc)
            )
        else:
            logger.error("Failed to parse Ontology.")

    def load_stage1_context(self, stage1_data: Dict[str, Any]):
        """Build label→id lookup and prompt summary from Stage 1 output."""
        devices = stage1_data.get('devices', stage1_data.get('instances', []))
        summary_lines = []
        for device in devices:
            label = device.get('label', '')
            proto_id = device['id']
            device_class = device.get('ontology_class', device.get('class', '?'))
            if label:
                self._label_to_id[label] = proto_id
            summary_lines.append(f"- ID: {proto_id} | Label: {label} | Class: {device_class}")
        self.existing_prototypes_summary = "\n".join(summary_lines)
        logger.info(f"Loaded {len(devices)} Stage 1 prototypes into label→id lookup")

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for VLM input"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _build_prompt(self, context_before: List[Dict], context_after: List[Dict]) -> str:
        """
        Build extraction prompt with TBox and ABox context.
        
        CHANGES IN v2.0:
        - Updated terminology (Instance → Prototype)
        - Enhanced guidance for variant detection
        - Clearer instructions for prototype vs instance distinction
        """
        # Prepare context strings
        c_before = "\n".join([f"  [{e['element_type']}]: {e['text'][:200]}" for e in context_before]) or "(None)"
        c_after = "\n".join([f"  [{e['element_type']}]: {e['text'][:200]}" for e in context_after]) or "(None)"
        
        return f"""
## Role
You are a Technical Drawing Expert enriching a Knowledge Graph by analyzing engineering drawings with their surrounding document context.

**IMPORTANT CONTEXT - Prototype vs Instance**:
You are identifying device PROTOTYPES (abstract families) and dimension PROTOTYPES, not specific instances.
- A PROTOTYPE represents a device model/family or dimension parameter
- INSTANCES will be created later in Stage 3 when table data provides specific values
- Example: "Type 3244 Valve" is ONE prototype, even if it has variants like DN15, DN20 (those are instances)
- Variants like "Type 3244 with Optional Flange" may be a separate prototype or enrichment of existing one

## STEP 0: Image Type Check (CRITICAL FIRST STEP)

**First, determine if this is a dimension-related engineering drawing.**

A dimension-related engineering drawing MUST contain:
- Technical/mechanical drawings showing device components
- **Dimension annotations**: dimension lines with arrows (←→, ↕)
- **Parameter labels**: H, H1, L, L1, DN, Ø, etc.
- Measurement values with units (mm, inch, etc.)

**SKIP if image is:**
- Product photograph (no dimension lines)
- Block diagram or flowchart
- Electrical schematic without mechanical dimensions
- Table or text-only content
- Logo, icon, or decorative image
- Installation scene or application photo

**Output Decision:**
- If NOT a dimension drawing → Set `is_dimension_drawing: false`, provide `skip_reason`, leave other fields empty
- If IS a dimension drawing → Set `is_dimension_drawing: true`, proceed with full extraction

---

## Task (Only if is_dimension_drawing = true)
Analyze the provided engineering drawing along with its contextual information to:
1. **Identify or enrich Device Prototype(s)** shown in the drawing
2. **Extract Dimension Prototypes** with their visual properties
3. **Link Dimension Prototypes to Device Prototypes**

## 1. TBox Constraint (Strict Schema)
Use ONLY the properties defined below for the 'Dimension' class.
{self.dimension_class_def}

## 2. ABox Context (Existing Device Prototypes from Stage 1)
These device prototypes already exist in the knowledge graph:
```text
{self.existing_prototypes_summary}
```

## 3. Document Context Around This Image

**Context BEFORE the image (reading order):**
```
{c_before}
```

**Context AFTER the image (reading order):**
```
{c_after}
```

## Inference and Extraction Rules (Only if is_dimension_drawing = true)

### Step 1: Device Prototype Identification & Discovery

Analyze the context (text before/after the image) to determine which device prototype(s) are being illustrated.

**Decision Logic:**

**Case A: Existing Prototype Enrichment**
- If the context mentions a device found in `Existing Device Prototypes from Stage 1` (e.g., ID `proto_valve_3241`):
  - Use that **Exact ID** in relationships
  - You may optionally add a `NewDevicePrototype` entry if you're adding NEW attributes not in Stage 1
  - Set `is_variant: false`

**Case B: New Prototype Discovery**
- If the context describes a NEW device model not in the existing list (e.g., "Type 5000"):
  1. Create a `NewDevicePrototype`.
  2. **Naming Rule**: Assign a **meaningful semantic temp_id** (e.g., "temp_valve_5000", "temp_actuator_series_2000").
  3. **CRITICAL**: Provide the EXACT `label` (e.g., "Type 5000") - this will be checked against the global registry.
  4. Set `is_variant: false` for a completely new device model
  5. Provide clear `inference_reasoning` explaining which context clues led to this discovery.

**Case C: Variant Discovery**
- If the context describes a VARIANT of an existing device (e.g., "Type 3244 with Extended Stem", "Type 3244-A"):
  1. Create a `NewDevicePrototype`.
  2. Use temp_id like "temp_valve_3244_variant_a"
  3. Set `is_variant: true`
  4. Label should include variant designation: "Type 3244 Variant A" or "Type 3244 with Extended Stem"
  5. Explain in `inference_reasoning` how this differs from base model

**Important Principle**:
- ONE prototype per device family (e.g., "Type 3244")
- Variants that differ significantly may be separate prototypes
- Use context clues to decide: is this a minor variation (same prototype) or significant variant (new prototype)?

### Step 2: Dimension Prototype Extraction

For each dimension annotation visible in the drawing:
1. **Identify the parameter** (H, H1, L, L1, Ø, d, DN, etc.)
2. **Extract PROTOTYPE-level properties** (NOT specific values):
   - `parameter_name`: The dimension parameter label exactly as shown (e.g., "H1", "L", "DN")
   - `dimension_type`: Classify the line orientation using **exactly one** of the four allowed values:
     - `"vertical"` — dimension line runs top-to-bottom (height, elevation parameters)
     - `"horizontal"` — dimension line runs left-to-right (length, width parameters)
     - `"diameter"` — circular/radial measurement (Ø, DN, d parameters)
     - `"other"` — diagonal, angular, or any orientation that does not fit above
   - DO NOT include specific numeric values
3. Assign a `temp_id` to each dimension prototype (e.g., "dim_h1_3244", "dim_length_5000").

### Step 3: Linking

Create relationships connecting:
- Each `DimensionPrototype` (object) to its owner `DevicePrototype` (subject).
- **subject_temp_id**: Use either:
  - The **Existing Stage 1 ID** (from Case A) like "proto_valve_3241", OR
  - Your **Semantic Temp ID** (from Case B/C) like "temp_valve_5000"
- **predicate**: `hasDimension`
- **object_temp_id**: Your dimension prototype temp_id

## Output Requirements
1. **is_dimension_drawing**: true/false (ALWAYS set this first)
2. **context_analysis**: Brief summary of how you identified the device prototype.
3. **new_device_prototypes**: List of NEW or VARIANT prototypes discovered (may be empty if only enriching existing)
4. **dimension_prototypes**: All dimension prototypes found in the image; only `parameter_name` and `dimension_type` are required
5. **relationships**: Links connecting device prototypes to dimension prototypes

## Important Notes
1. **Check Image Type FIRST**: Don't waste effort extracting from non-technical images.
2. **Trust the Context**: The text before/after the image is your primary source for identifying which device this drawing represents.
3. **Strict dimension_type values**: Use only `"vertical"`, `"horizontal"`, `"diameter"`, or `"other"` — no other strings accepted.
4. **Be Comprehensive**: Extract ALL visible dimension parameters.
5. **Flag New Prototypes**: Do propose new device prototypes if context is clear, but be careful with variants vs new models.
6. **Prototype Level**: Remember you're extracting PROTOTYPES (parameters, orientations), not INSTANCES (specific values).
"""

    def _generate_dimension_prototype_id(self, device_id: str, parameter_name: str) -> str:
        """
        Generate deterministic dimension prototype ID.
        
        Format: {device_id}_dim_{sanitized_parameter_name}
        Example: "proto_valve_3244_dim_h1"
        
        NEW in v2.0: Deterministic generation for better traceability
        """
        sanitized_param = re.sub(r'[^\w]', '_', parameter_name.lower())
        return f"{device_id}_dim_{sanitized_param}"

    def process_image(
        self, 
        image_path: str, 
        context_before: List[Dict], 
        context_after: List[Dict]
    ) -> Optional[Dict]:
        """
        Process a single image to extract device and dimension prototypes.
        
        CHANGES IN v2.0:
        - Uses GlobalIdentityRegistry for device ID resolution
        - Generates deterministic dimension prototype IDs
        - Registers dimension prototypes to registry
        - Returns structured data with 'devices' and 'dimensions' keys
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None

        b64_img = self._encode_image(image_path)
        prompt = self._build_prompt(context_before, context_after)
        
        try:
            patch: VisualGraphPatch = self.llm.invoke([HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
            ])])
            
            if not patch.is_dimension_drawing:
                logger.debug(f"Skipped non-dimension drawing: {patch.skip_reason}")
                return None

            # =========================================================
            # MIDDLEWARE: temp_id → final_id resolution
            # =========================================================

            id_map: Dict[str, str] = {}
            resolved_device_prototypes = []
            resolved_dimension_prototypes = []

            # 1. Resolve Device Prototypes against Stage 1 label→id lookup
            for new_dev in patch.new_device_prototypes:
                existing_id = self._label_to_id.get(new_dev.label)
                if existing_id:
                    # Already known from Stage 1: reuse its ID
                    final_device_id = existing_id
                    id_map[new_dev.temp_id] = final_device_id
                    logger.info(f"  Matched Stage 1 prototype: '{new_dev.label}' → {final_device_id}")
                else:
                    # Genuinely new device discovered in Stage 2
                    slug = re.sub(r'\W+', '_', new_dev.label.lower()).strip('_')
                    final_device_id = f"proto_{slug}_{uuid.uuid4().hex[:6]}"
                    id_map[new_dev.temp_id] = final_device_id
                    self._label_to_id[new_dev.label] = final_device_id  # prevent duplicates within batch
                    dev_dict = new_dev.model_dump()
                    dev_dict['id'] = final_device_id
                    dev_dict['is_new_discovery'] = True
                    del dev_dict['temp_id']
                    resolved_device_prototypes.append(dev_dict)
                    logger.info(f"  New device prototype: '{new_dev.label}' → {final_device_id}")

            # 2. Resolve Dimension Prototypes (deterministic IDs from parent + parameter)
            for dim in patch.dimension_prototypes:
                parent_device_id = None
                for rel in patch.relationships:
                    if rel.object_temp_id == dim.temp_id:
                        parent_device_id = id_map.get(rel.subject_temp_id, rel.subject_temp_id)
                        break

                if not parent_device_id:
                    logger.warning(f"Dimension '{dim.parameter_name}' has no parent device, skipping")
                    continue

                final_dim_id = self._generate_dimension_prototype_id(parent_device_id, dim.parameter_name)
                id_map[dim.temp_id] = final_dim_id

                dim_dict = dim.model_dump()
                dim_dict['id'] = final_dim_id
                dim_dict['belongs_to_device'] = parent_device_id
                dim_dict['ontology_class'] = 'Dimension'
                del dim_dict['temp_id']
                resolved_dimension_prototypes.append(dim_dict)
                logger.debug(f"  Dimension prototype: {dim.parameter_name} → {final_dim_id}")

            # 3. Rewire Relationships with final IDs
            resolved_relationships = []
            for rel in patch.relationships:
                # Resolve both subject and object
                final_subj = id_map.get(rel.subject_temp_id, rel.subject_temp_id)
                final_obj = id_map.get(rel.object_temp_id, rel.object_temp_id)

                rel_dict = rel.model_dump()
                rel_dict['subject_id'] = final_subj
                rel_dict['object_id'] = final_obj
                del rel_dict['subject_temp_id']
                del rel_dict['object_temp_id']
                resolved_relationships.append(rel_dict)

            return {
                "devices": resolved_device_prototypes,  # NEW: changed from new_devices
                "dimensions": resolved_dimension_prototypes,  # NEW: changed from instances
                "relationships": resolved_relationships,
                "analysis": patch.context_analysis
            }

        except Exception as e:
            logger.error(f"Error processing image {os.path.basename(image_path)}: {e}", exc_info=True)
            return None

# =====================================================
# Main Execution
# =====================================================

def execute_stage2_visual_extraction(
    stage1_data: Dict,
    images_json_path: str,
    ontology_path: str,
    api_key: str,
    output_dir: str,
    doc_name: str = "",
    model_name: str = "gpt-4o",
) -> Dict:
    """
    Main Stage 2 execution function.

    Args:
        stage1_data: Stage 1 result dict (passed directly from chain).
        images_json_path: Path to images JSON.
        ontology_path: Path to TBox ontology.
        api_key: OpenAI API key.
        output_dir: Directory for output files.
        doc_name: Document name for output file naming. Derived from stage1_data if omitted.
        model_name: LLM model name.
    """
    logger.info("="*70)
    logger.info("STAGE 2: VISUAL EXTRACTION (v2.0)")
    logger.info("="*70)

    # 1. Load images
    with open(images_json_path, 'r', encoding='utf-8') as f:
        images_data = json.load(f)
    images_list = images_data.get('images', []) if isinstance(images_data, dict) else images_data

    if not doc_name:
        doc_name = stage1_data.get('_metadata', {}).get('doc', 'unknown')

    stage1_devices = stage1_data.get('devices', stage1_data.get('instances', []))
    logger.info(f"Stage 1 prototypes loaded: {len(stage1_devices)}")

    # 2. Init Extractor
    extractor = Stage2VisualExtractor(api_key, ontology_path, model_name=model_name)
    extractor.load_stage1_context(stage1_data)

    all_new_devices = []
    all_new_dims = []
    all_new_rels = []

    # 4. Process Images
    for idx, img_archive in enumerate(images_list, 1):
        img_path = img_archive.get('cropped_image_path')
        if not img_path:
            logger.warning(f"[{idx}/{len(images_list)}] Missing image path, skipping")
            continue
        
        logger.info(f"[{idx}/{len(images_list)}] Processing {os.path.basename(img_path)}...")
        
        patch_result = extractor.process_image(
            img_path, 
            img_archive.get('context_before', []), 
            img_archive.get('context_after', [])
        )
        
        if patch_result:
            logger.info(
                f"  → Found {len(patch_result['devices'])} device prototypes, "
                f"{len(patch_result['dimensions'])} dimension prototypes"
            )
            all_new_devices.extend(patch_result['devices'])
            all_new_dims.extend(patch_result['dimensions'])
            all_new_rels.extend(patch_result['relationships'])
        else:
            logger.info("  → Skipped (Not a dimension drawing)")

    # 5. Merge with Stage 1
    merged_devices = stage1_devices + all_new_devices + all_new_dims
    merged_relationships = stage1_data.get('relationships', []) + all_new_rels
    
    # Save Result
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"{doc_name}_stage2_enriched_graph.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump({
            "devices": merged_devices,  # NEW: unified field name
            "relationships": merged_relationships,
            "_metadata": {
                "stage": 2, 
                "version": "2.0",
                "structure": "enriched_prototypes",
                "total_devices": len(merged_devices),
                "new_device_prototypes": len(all_new_devices),
                "dimension_prototypes": len(all_new_dims)
            }
        }, f, indent=2, ensure_ascii=False)
        
    # =========================================================
    # Final Reporting
    # =========================================================
    logger.info(f"\n{'='*70}")
    logger.info("STAGE 2 COMPLETE: VISUAL EXTRACTION SUMMARY")
    logger.info(f"{'='*70}")

    # Report 1: New Device Prototypes
    if all_new_devices:
        logger.info(f"\n[+] NEW DEVICE PROTOTYPES DISCOVERED: {len(all_new_devices)}")
        for dev in all_new_devices:
            variant_tag = " [VARIANT]" if dev.get('is_variant') else ""
            logger.info(f"  [{dev['id']}] {dev['ontology_class']} ({dev['label']}){variant_tag}")
            reasoning_short = dev.get('inference_reasoning', 'N/A')
            if len(reasoning_short) > 100:
                reasoning_short = reasoning_short[:97] + "..."
            logger.info(f"      Reasoning: {reasoning_short}")
    else:
        logger.info("\n[+] No new device prototypes discovered from images.")

    # Report 2: Dimension Prototypes
    logger.info(f"\n[+] DIMENSION PROTOTYPES EXTRACTED: {len(all_new_dims)}")
    logger.info(f"[+] RELATIONSHIPS FORMED:           {len(all_new_rels)}")

    # Report 3: Global Graph Stats
    logger.info(f"\n[=] ENRICHED GRAPH STATUS")
    logger.info(f"  Total Device Prototypes:    {len(merged_devices)}")
    logger.info(f"  Total Relationships:        {len(merged_relationships)}")
    
    logger.info(f"\nResult saved to: {final_path}")
    logger.info("="*70 + "\n")

    return {
        "devices": merged_devices,
        "relationships": merged_relationships,
        "_metadata": {
            "stage": 2,
            "version": "2.0",
            "structure": "enriched_prototypes",
            "doc": doc_name,
            "output_path": final_path,
            "total_devices": len(merged_devices),
            "new_device_prototypes": len(all_new_devices),
            "dimension_prototypes": len(all_new_dims)
        }
    }

if __name__ == "__main__":
    # NOTE: For end-to-end testing use src/chains/stage12_extraction_chain.py.
    # This standalone mode requires a Stage 1 JSON already saved to disk.
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

    doc_name = "SS_03"
    base_dir = project_root / "data" / "test_intermediate_results_2" / doc_name
    s1_path = base_dir / f"{doc_name}_stage1_skeleton.json"
    images_path = project_root / "data" / "test_intermediate_results" / doc_name / "images" / f"{doc_name}_images.json"
    ontology_path = project_root / "data" / "ontology" / "DeviceDimension_demo.rdf"

    if not s1_path.exists():
        logger.error(f"Stage 1 result not found: {s1_path}")
        sys.exit(1)

    with open(s1_path, 'r') as f:
        stage1_data = json.load(f)

    if images_path.exists():
        execute_stage2_visual_extraction(
            stage1_data=stage1_data,
            images_json_path=str(images_path),
            ontology_path=str(ontology_path),
            api_key=api_key,
            output_dir=str(base_dir),
            doc_name=doc_name,
        )
    else:
        logger.error(f"Images file not found: {images_path}")