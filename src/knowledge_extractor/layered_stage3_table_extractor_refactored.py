# src/knowledge_extractor/layered_stage3_table_extractor_refactored.py

"""
Stage 3: Instance Fission (Table + LLM) - Two-Layer Architecture

VERSION HISTORY:
- v3.0 (2025-05-25): Two-Layer Architecture Refactor
  * OUTPUT: prototype_graph (subclass layer) + instance_graph (ABox)
  * Prototype layer: each device prototype becomes a subclass of its most-specific
    TBox ontology_class (as determined by Stage 1/2 classification). Inter-prototype
    object properties (hasActuator, hasBodySeat, etc.) are stored here and inherited
    by all instances implicitly — never duplicated per instance.
  * Instance layer: one device instance per table row, carrying only TBox-defined
    data properties (dimension values/units per variant key).
  * Removed: _inherit_relationships() — object properties stay at prototype level
  * Removed: _map_column_to_property() — non-dimension table columns are ignored
    at the instance level (prototype attributes already captured in Stage 1/2)
  * Fixed: dimension_type enum aligned with Stage 2: {vertical, horizontal, diameter, other}
  * Cleaned: dead anchor inheritance code from _clone_dimension()

- v2.1 (2025-01-16): Table-Only Dimension Creation
  * NEW: Extract ALL dimension columns from table (not just those with prototypes)
  * ENHANCED: _is_dimension_column() with comprehensive pattern matching

- v2.0 (2025-01-16): Major Refactoring
  * Terminology Update: Prototypes → Instances
  * Parameter-Based Fission, GlobalIdentityRegistry Integration

ARCHITECTURE (v3.0):
                        ┌─ prototype_graph ──────────────────────────────────────┐
  Stage 2 prototypes:   │  GlobeValve                                            │
  proto_valve_3244  ────►    └─ Type3244  ──hasActuator──► Type3271              │
  (GlobeValve)          │         dim_prototypes: [H1(vertical), L(horizontal)]  │
                        │  Actuator                                               │
  proto_actuator_3271 ──►    └─ Type3271                                         │
  (Actuator)            └────────────────────────────────────────────────────────┘
                                          ↓ fission (one row = one instance)
                        ┌─ instance_graph ───────────────────────────────────────┐
                        │  Type3244_DN15  instance_of: proto_valve_3244          │
                        │    dimensions: [{H1, vertical, 120.0, mm}, ...]        │
                        │  Type3244_DN20  instance_of: proto_valve_3244          │
                        │    dimensions: [{H1, vertical, 150.0, mm}, ...]        │
                        └────────────────────────────────────────────────────────┘

  Object properties (hasActuator etc.) live at prototype level — no per-instance copies.

Input:
  - stage2_enriched_graph.json
  - tables.json
  - ontology RDF file

Output:
  - {basename}_stage3_instances.json
    {
      "prototype_graph": {
        "device_prototypes": [
          {
            "id": "proto_valve_3244",
            "ontology_class": "GlobeValve",
            "label": "Type 3244",
            "attributes": [...],
            "dimension_prototypes": [
              {"parameter_name": "H1", "dimension_type": "vertical"},
              {"parameter_name": "L",  "dimension_type": "horizontal"}
            ]
          }
        ],
        "object_relations": [
          {"subject_id": "proto_valve_3244", "predicate": "hasActuator", "object_id": "proto_actuator_3271"}
        ]
      },
      "instance_graph": {
        "device_instances": [
          {
            "id": "proto_valve_3244_DN15",
            "prototype_id": "proto_valve_3244",
            "ontology_class": "GlobeValve",
            "label": "Type 3244 - DN15",
            "variant_key": "DN15",
            "dimensions": [
              {"parameter_name": "H1", "dimension_type": "vertical",   "value": 120.0, "unit": "mm"},
              {"parameter_name": "L",  "dimension_type": "horizontal",  "value": 85.0,  "unit": "mm"}
            ]
          }
        ]
      },
      "_metadata": {...}
    }
"""

import os
import sys
import json
import logging
import uuid
import re
from collections import defaultdict
from typing import Dict, Any, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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
# TBox Schema Mapper (Dynamic Resolution)
# =====================================================

class TBoxSchemaMap:
    """
    Resolves Dimension class property names from the loaded TBox.
    Used to identify which Stage 2 relations are hasDimension links
    and to build TBox-compliant dimension data in instances.
    """

    def __init__(self, loos: Optional[OntologyLOOS]):
        self.dim_class = "Dimension"
        self.rel_has_dim = "hasDimension"
        self.prop_val = "hasDimensionValue"
        self.prop_unit = "hasDimensionUnit"
        self.prop_param = "hasDimensionParameterName"
        self.valid_data_props: List[str] = []

        if loos:
            self._resolve_schema(loos)

    def _resolve_schema(self, loos: OntologyLOOS):
        self.valid_data_props = list(loos.datatype_properties.keys())

        # 1. Dimension class
        candidates = [c for c in loos.classes if "Dimension" in c]
        if candidates:
            self.dim_class = candidates[0]
        else:
            logger.warning(f"Dimension class not found in TBox, using default: {self.dim_class}")

        # 2. hasDimension object property (range contains Dimension)
        found_rel = False
        for prop, info in loos.object_properties.items():
            if self.dim_class in info.range:
                self.rel_has_dim = prop
                found_rel = True
                break
        if not found_rel:
            logger.warning(f"hasDimension not found in TBox, using default: {self.rel_has_dim}")

        # 3. Dimension data properties
        dim_props = [
            (p, info) for p, info in loos.datatype_properties.items()
            if self.dim_class in info.domain or not info.domain
        ]
        found_val = found_unit = found_param = False
        for prop, info in dim_props:
            pl = prop.lower()
            if not found_val and ("value" in pl or "val" in pl) and ("dimension" in pl or "dim" in pl):
                self.prop_val = prop;  found_val = True
            if not found_unit and "unit" in pl:
                self.prop_unit = prop; found_unit = True
            if not found_param and ("name" in pl or "param" in pl) and ("dimension" in pl or "parameter" in pl):
                self.prop_param = prop; found_param = True

        for label, val, found in [
            ("hasDimensionValue",         self.prop_val,   found_val),
            ("hasDimensionUnit",          self.prop_unit,  found_unit),
            ("hasDimensionParameterName", self.prop_param, found_param),
        ]:
            if not found:
                logger.warning(f"⚠️  {label} not found in TBox, using hardcoded: {val}")

        logger.info(
            f"TBox resolved: dim='{self.dim_class}', rel='{self.rel_has_dim}', "
            f"val='{self.prop_val}', unit='{self.prop_unit}', param='{self.prop_param}'"
        )


# =====================================================
# Dimension Fission Strategy
# =====================================================

class DimensionFissionStrategy:
    """
    Fissions dimension prototypes + table rows into concrete dimension entries.

    v3.0: Dimensions are embedded directly in the device instance dict
    (no separate dimension entity objects or hasDimension relationship dicts).
    The TTL conversion layer is responsible for creating Dimension individuals.
    """

    def __init__(self, tbox_map: TBoxSchemaMap):
        self.tbox_map = tbox_map

    def fission_device(
        self,
        device_prototype: Dict,
        dimension_prototypes: List[Dict],
        table_row: Dict,
        variant_key: str,
    ) -> Dict:
        """
        Create one device instance from a single table row.

        Algorithm:
        1. For each Stage 2 dimension prototype: look up its value in the table row.
        2. For remaining dimension columns in the table without a prototype:
           create a dimension entry from the table value alone.

        Returns a flat device instance dict with an embedded 'dimensions' list.
        Object properties (hasActuator etc.) are NOT included here —
        they are preserved at the prototype level in prototype_graph.object_relations.
        """
        proto_id = device_prototype['id']
        safe_key = re.sub(r'[^\w]', '_', variant_key)
        instance_id = f"{proto_id}_{safe_key}"

        dimensions: List[Dict] = []
        processed_params: set = set()

        # Step 1: Stage 2 dimension prototypes — inherit type, look up value
        for dim_proto in dimension_prototypes:
            param_name = dim_proto.get('parameter_name', '')
            if not param_name:
                continue
            parsed = self._find_dimension_value_in_row(table_row, param_name)
            entry: Dict[str, Any] = {
                "parameter_name": param_name,
                "dimension_type": dim_proto.get('dimension_type', 'other'),
            }
            if parsed:
                entry[self.tbox_map.prop_val]  = parsed["value"]
                entry[self.tbox_map.prop_unit] = parsed["unit"]
            dimensions.append(entry)
            processed_params.add(self._normalize(param_name))

        # Step 2: Table-only dimension columns (no Stage 2 prototype)
        for col_name, col_value in table_row.items():
            normalized = self._normalize(col_name)
            if self._is_dimension_column(col_name) and normalized not in processed_params:
                parsed = self._parse_value(col_value)
                if parsed:
                    dimensions.append({
                        "parameter_name": normalized,
                        "dimension_type": self._infer_type(normalized),
                        self.tbox_map.prop_val:  parsed["value"],
                        self.tbox_map.prop_unit: parsed["unit"],
                    })
                    processed_params.add(normalized)

        return {
            "id": instance_id,
            "prototype_id": proto_id,
            "ontology_class": device_prototype.get('ontology_class', 'Device'),
            "label": f"{device_prototype.get('label', 'Unknown')} - {variant_key}",
            "variant_key": variant_key,
            "dimensions": dimensions,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, name: str) -> str:
        """Strip unit hints and non-alphanumeric chars, lowercase."""
        s = re.sub(r'\s*\([^)]*\)', '', name)   # remove "(mm)", "(kg)" etc.
        return re.sub(r'[^\w]', '', s).lower()

    def _find_dimension_value_in_row(self, row: Dict, param: str) -> Optional[Dict]:
        """Exact → case-insensitive → substring match."""
        if param in row:
            return self._parse_value(row[param])
        pl = param.lower()
        for col, val in row.items():
            if col.lower() == pl:
                return self._parse_value(val)
        for col, val in row.items():
            if pl in col.lower():
                return self._parse_value(val)
        return None

    def _parse_value(self, raw: Any) -> Optional[Dict]:
        """Parse raw table cell into {value: float, unit: str}."""
        if raw is None or str(raw).strip() == "":
            return None
        if isinstance(raw, (int, float)):
            return {"value": float(raw), "unit": "mm"}
        s = str(raw)
        unit = "mm"
        m = re.search(r'(mm|cm|m|inch|in|kg|g|bar|psi)', s, re.IGNORECASE)
        if m:
            unit = m.group(1).lower()
            s = s[:m.start()]
        s = re.sub(r'[Øø∅φΦ]', '', s)
        s = re.sub(r'[^\d.-]', '', s)
        try:
            v = float(s) if s else None
            return {"value": v, "unit": unit} if v is not None else None
        except ValueError:
            logger.warning(f"Cannot parse dimension value: {raw}")
            return None

    def _infer_type(self, param: str) -> str:
        """
        Infer dimension_type from parameter name.
        Enum: vertical | horizontal | diameter | other
        """
        p = param.lower().strip()
        if re.match(r'^d\d*$', p) or re.match(r'^dn\d*$', p) or any(x in p for x in ['diameter', 'radius']):
            return "diameter"
        if re.match(r'^h\d*$', p) or 'height' in p or 'vertical' in p:
            return "vertical"
        if re.match(r'^[lbw]\d*$', p) or any(x in p for x in ['length', 'width', 'horizontal']):
            return "horizontal"
        return "other"

    def _is_dimension_column(self, col: str) -> bool:
        """Return True if the column header looks like a dimension parameter."""
        cl = col.lower()
        patterns = [
            r'^[hldwbt]\d*$',
            r'^dn\d*$', r'^pn\d*$', r'^kvs?\d*$', r'^cv\d*$',
            r'[øØ∅φΦ]',
            r'height', r'width', r'length', r'depth',
            r'diameter', r'radius', r'weight', r'mass',
            r'thickness', r'clearance', r'dimension', r'size',
        ]
        if any(re.search(pat, cl) for pat in patterns):
            return True
        unit_hints = ['(mm)', '(cm)', '(m)', '(kg)', '(g)', '(bar)', '(psi)',
                      '[mm]', '[cm]', '[m]', '[kg]', '[g]', '[bar]', '[psi]']
        return any(u in cl for u in unit_hints)


# =====================================================
# Pydantic Models (LLM Interface)
# =====================================================

class ExtractedDimensionColumn(BaseModel):
    column_name: str = Field(..., description="Exact column name from table header")
    unit: Optional[str] = Field(default="mm", description="Unit from header or inferred")


class TableFissionRow(BaseModel):
    variant_key: str = Field(..., description="Row differentiator, e.g. 'DN15', 'PN40'")
    row_data: Dict[str, Any] = Field(..., description="column_name → value (str/float/int/None)")

    @field_validator('row_data', mode='before')
    @classmethod
    def clean_row_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v, dict):
            return v
        cleaned: Dict[str, Any] = {}
        for key, value in v.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, dict):
                logger.warning(f"Column '{key}' has nested dict; extracting first value")
                cleaned[key] = next(iter(value.values()), None) if value else None
            elif isinstance(value, (str, float, int, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned


class NewPrototypeIntent(BaseModel):
    ontology_class: str = Field(..., description="Most-specific TBox class name")
    label: str = Field(..., description="Exact device family name")
    reasoning: str = Field(..., description="Why no existing prototype matched")


class TableAnalysisResult(BaseModel):
    is_dimensional_table: bool = Field(..., description="True if table contains device specs/dimensions")
    skip_reason: Optional[str] = Field(None)
    chosen_prototype_id: Optional[str] = Field(None, description="Matching prototype ID from Stage 2")
    new_prototype_intent: Optional[NewPrototypeIntent] = Field(None)
    dimension_columns: List[ExtractedDimensionColumn] = Field(default_factory=list)
    fission_rows: List[TableFissionRow] = Field(default_factory=list)


# =====================================================
# Core Logic: Stage 3 Table Extractor
# =====================================================

class Stage3TableExtractor:
    """
    Stage 3 Table Extractor — Two-Layer Architecture (v3.0).

    prototype_graph: structural layer preserving Stage 1/2 device prototypes as
    subclasses of their TBox ontology_class, with inter-prototype object properties.

    instance_graph: ABox instances (one per table row) with embedded dimension data.
    """

    def __init__(self, api_key: str, ontology_path: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=4000,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        loader = OntologyLoaderOwlready2(ontology_path)
        if not loader.load():
            logger.error(f"Failed to load ontology: {ontology_path}")
        loader.extract_structure()

        self.tbox_map = TBoxSchemaMap(loader.loos)
        self.fission = DimensionFissionStrategy(self.tbox_map)

        # Populated by load_stage2_context()
        self._device_prototypes: Dict[str, Dict] = {}          # proto_id → prototype dict
        self._dim_protos_by_device: Dict[str, List[Dict]] = {} # proto_id → [dim_proto dicts]
        self._label_to_id: Dict[str, str] = {}                 # label → proto_id
        self._prototype_object_relations: List[Dict] = []      # non-hasDimension rels from Stage 2

        logger.info("✓ Stage 3 Table Extractor initialized (v3.0)")

    # ------------------------------------------------------------------
    # Stage 2 Context Loading
    # ------------------------------------------------------------------

    def load_stage2_context(self, stage2_data: Dict[str, Any]):
        """
        Index Stage 2 output into internal lookup structures.

        Separates:
        - Device prototypes (non-Dimension ontology_class) → _device_prototypes
        - Dimension prototypes → _dim_protos_by_device (keyed by belongs_to_device)
        - Object relations between prototypes (non-hasDimension) → _prototype_object_relations
          These are the structural device-device properties that stay at prototype level.
        - hasDimension relations → consumed internally via belongs_to_device field; not output.
        """
        all_entities = stage2_data.get('devices', stage2_data.get('instances', []))
        all_relations = stage2_data.get('relationships', [])

        dim_by_device: Dict[str, List] = defaultdict(list)

        for entity in all_entities:
            if entity.get('ontology_class') == self.tbox_map.dim_class:
                parent = entity.get('belongs_to_device')
                if parent:
                    dim_by_device[parent].append(entity)
            else:
                proto_id = entity['id']
                self._device_prototypes[proto_id] = entity
                label = entity.get('label', '')
                if label:
                    self._label_to_id[label] = proto_id

        self._dim_protos_by_device = dict(dim_by_device)

        # Keep only structural object properties at prototype level.
        # hasDimension links are redundant here (already indexed via belongs_to_device).
        self._prototype_object_relations = [
            r for r in all_relations
            if r.get('predicate') != self.tbox_map.rel_has_dim
        ]

        dim_total = sum(len(v) for v in dim_by_device.values())
        logger.info(
            f"Stage 2 loaded: {len(self._device_prototypes)} device prototypes, "
            f"{dim_total} dim prototypes, "
            f"{len(self._prototype_object_relations)} prototype object relations"
        )

    # ------------------------------------------------------------------
    # LLM Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, table_data: Dict, c_before: str, c_after: str) -> str:
        candidates_str = "\n".join(
            f"- ID: {pid} | Label: {p.get('label', '?')} | Class: {p.get('ontology_class', '?')}"
            for pid, p in self._device_prototypes.items()
        ) or "(No device prototypes from Stage 2)"

        table_json_str = json.dumps(table_data, indent=2, ensure_ascii=False)
        valid_props = ", ".join(self.tbox_map.valid_data_props) or "see TBox"

        return f"""
## Role
You are a Datasheet Specialist extracting device instances from specification tables.

## Context
Stage 1+2 identified device PROTOTYPES (e.g. "Type 3244 GlobeValve").
Stage 3 creates INSTANCES — one per table row (e.g. "Type 3244 DN15", "Type 3244 DN20").
Instances carry only dimension data values. Structural properties (hasActuator etc.)
are already stored at the prototype level and must NOT be extracted here.

## STEP 0: Relevance Check
Set is_dimensional_table: true  → device specification table (dimension columns: H, L, DN, weight…)
Set is_dimensional_table: false → revision history, troubleshooting, ordering codes without specs

## Task (if relevant)

### 1. Identify Prototype
Match the table to an existing prototype using document context.
Use chosen_prototype_id if a match is found; otherwise fill new_prototype_intent.

### 2. Extract Dimension Columns
List ALL columns that represent dimension parameters (exact name + unit).

### 3. Extract Fission Rows
One entry per table row:
  - variant_key: unique human-readable differentiator (e.g. "DN15", "DN20_PN40")
  - row_data: complete column→value mapping with EXACT column names

## Available Device Prototypes (from Stage 2)
```
{candidates_str[:5000]}
```

## Document Context
Before table:
{c_before[:1000]}

After table:
{c_after[:1000]}

Table content:
```json
{table_json_str[:6000]}
```

## Output (JSON only, no markdown)
{{
  "is_dimensional_table": true,
  "skip_reason": null,
  "chosen_prototype_id": "proto_valve_3244",
  "new_prototype_intent": null,
  "dimension_columns": [
    {{"column_name": "H1", "unit": "mm"}},
    {{"column_name": "L",  "unit": "mm"}}
  ],
  "fission_rows": [
    {{"variant_key": "DN15", "row_data": {{"DN": 15, "H1": 120.0, "L": 85.0}}}},
    {{"variant_key": "DN20", "row_data": {{"DN": 20, "H1": 150.0, "L": 95.0}}}}
  ]
}}

Valid TBox data properties for reference: {valid_props}
"""

    # ------------------------------------------------------------------
    # Table Processing
    # ------------------------------------------------------------------

    def process_table(self, table: Dict, table_idx: int) -> Optional[Dict]:
        """Process one table: LLM analysis → prototype resolution → instance fission."""
        c_before = "\n".join(
            f"[{e.get('element_type', '?')}]: {e.get('text', '')[:200]}"
            for e in table.get('context_before', [])
        ) or "(None)"
        c_after = "\n".join(
            f"[{e.get('element_type', '?')}]: {e.get('text', '')[:200]}"
            for e in table.get('context_after', [])
        ) or "(None)"

        try:
            response = self.llm.invoke([HumanMessage(content=self._build_prompt(table, c_before, c_after))])
            json_str = response.content if hasattr(response, 'content') else str(response)
            result = TableAnalysisResult(**json.loads(json_str))
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error on Table #{table_idx}: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM error on Table #{table_idx}: {e}", exc_info=True)
            return None

        if not result.is_dimensional_table:
            return {"skipped": True, "reason": result.skip_reason}

        # ---- Resolve prototype ----
        final_proto_id: Optional[str] = None
        discovery_info: Optional[Dict] = None

        if result.chosen_prototype_id and result.chosen_prototype_id in self._device_prototypes:
            final_proto_id = result.chosen_prototype_id

        elif result.new_prototype_intent:
            intent = result.new_prototype_intent
            final_proto_id = self._label_to_id.get(intent.label)
            if not final_proto_id:
                slug = re.sub(r'\W+', '_', intent.label.lower()).strip('_')
                final_proto_id = f"proto_{slug}_{uuid.uuid4().hex[:6]}"
                self._label_to_id[intent.label] = final_proto_id
                new_proto = {
                    "id": final_proto_id,
                    "label": intent.label,
                    "ontology_class": intent.ontology_class,
                    "attributes": [],
                    "is_new_discovery": True,
                }
                self._device_prototypes[final_proto_id] = new_proto
                discovery_info = {
                    "id": final_proto_id, "label": intent.label,
                    "class": intent.ontology_class, "reasoning": intent.reasoning,
                }
                logger.info(f"  New prototype (Stage 3): '{intent.label}' → {final_proto_id}")
            else:
                logger.info(f"  Matched prototype by label: '{intent.label}' → {final_proto_id}")

        if not final_proto_id:
            return {"skipped": True, "reason": "No valid prototype resolved"}

        device_prototype  = self._device_prototypes[final_proto_id]
        dim_prototypes    = self._dim_protos_by_device.get(final_proto_id, [])
        if not dim_prototypes:
            logger.warning(f"  No Stage 2 dim prototypes for {final_proto_id} — table-only fission")

        # ---- Fission rows ----
        device_instances: List[Dict] = []
        for row in result.fission_rows:
            instance = self.fission.fission_device(
                device_prototype=device_prototype,
                dimension_prototypes=dim_prototypes,
                table_row=row.row_data,
                variant_key=row.variant_key,
            )
            device_instances.append(instance)
            logger.debug(f"    Instance: {instance['id']} ({len(instance['dimensions'])} dims)")

        return {
            "skipped": False,
            "device_instances": device_instances,
            "discovery": discovery_info,
            "instance_count": len(device_instances),
        }

    # ------------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        stage2_data: Dict,
        tables_path: str,
        output_dir: str,
        doc_name: str = "",
    ) -> Dict:
        """
        Main Stage 3 execution.

        Args:
            stage2_data:  Stage 2 result dict (passed directly from chain).
            tables_path:  Path to tables JSON file.
            output_dir:   Directory for output files.
            doc_name:     Document name; derived from metadata if omitted.

        Returns:
            Result dict with prototype_graph, instance_graph, _metadata.
        """
        logger.info("=" * 70)
        logger.info("STAGE 3: INSTANCE FISSION (v3.0)")
        logger.info("=" * 70)

        if not doc_name:
            doc_name = stage2_data.get('_metadata', {}).get('doc', 'unknown')

        with open(tables_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)

        self.load_stage2_context(stage2_data)

        all_instances: List[Dict] = []
        stats = {
            "total_tables": 0, "skipped": 0, "processed": 0,
            "new_prototypes": [], "total_instances": 0, "skip_reasons": {},
        }

        tables_list = tables_data.get('tables', [])
        stats["total_tables"] = len(tables_list)

        for idx, tbl in enumerate(tables_list, 1):
            logger.info(f"\n[{idx}/{len(tables_list)}] Table #{idx}")
            res = self.process_table(tbl, idx)

            if not res:
                stats["skipped"] += 1
                continue

            if res.get("skipped"):
                reason = res.get("reason", "unknown")
                logger.info(f"  → [SKIP] {reason}")
                stats["skipped"] += 1
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            else:
                count = res["instance_count"]
                logger.info(f"  → [OK] {count} device instances")
                all_instances.extend(res["device_instances"])
                stats["processed"] += 1
                stats["total_instances"] += count
                if res.get("discovery"):
                    d = res["discovery"]
                    logger.info(f"     [NEW PROTO] {d['label']} ({d['id']})")
                    stats["new_prototypes"].append(d)

        # ---- Build prototype_graph ----
        # Enrich each device prototype with its nested dimension prototypes.
        enriched_prototypes = []
        for proto_id, proto in self._device_prototypes.items():
            entry = {k: v for k, v in proto.items() if k != 'dimension_prototypes'}
            entry["dimension_prototypes"] = self._dim_protos_by_device.get(proto_id, [])
            enriched_prototypes.append(entry)

        # ---- Assemble result ----
        result = {
            "prototype_graph": {
                "device_prototypes": enriched_prototypes,
                "object_relations": self._prototype_object_relations,
            },
            "instance_graph": {
                "device_instances": all_instances,
            },
            "_metadata": {
                "stage": 3,
                "version": "3.0",
                "doc": doc_name,
                "total_prototypes": len(enriched_prototypes),
                "total_instances": stats["total_instances"],
                "tables_processed": stats["processed"],
                "tables_skipped": stats["skipped"],
                "new_prototypes_discovered": len(stats["new_prototypes"]),
            },
        }

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{doc_name}_stage3_instances.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # ---- Final report ----
        logger.info(f"\n{'=' * 70}")
        logger.info("STAGE 3 COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Device Prototypes:      {len(enriched_prototypes)}")
        logger.info(f"  Prototype Object Rels:  {len(self._prototype_object_relations)}")
        logger.info(f"  Device Instances:       {stats['total_instances']}")
        logger.info(f"  New Prototypes:         {len(stats['new_prototypes'])}")
        if stats['new_prototypes']:
            for p in stats['new_prototypes']:
                logger.info(f"    - {p['label']} ({p['id']})")
        logger.info(
            f"  Tables: {stats['total_tables']} total / "
            f"{stats['processed']} processed / {stats['skipped']} skipped"
        )
        if stats['skip_reasons']:
            for reason, cnt in stats['skip_reasons'].items():
                logger.info(f"    Skip '{reason}': {cnt}×")
        logger.info(f"\nResult saved to: {out_path}")
        logger.info("=" * 70 + "\n")

        result["_metadata"]["output_path"] = out_path
        return result


# =====================================================
# Main Entry Point
# =====================================================

if __name__ == "__main__":
    # NOTE: For end-to-end testing use src/chains/stage123_extraction_chain.py.
    # This standalone mode requires Stage 2 JSON already saved to disk.
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
    base_dir      = project_root / "data" / "test_intermediate_results_2" / doc_name
    stage2_file   = base_dir / f"{doc_name}_stage2_enriched_graph.json"
    tables_file   = project_root / "data" / "test_intermediate_results" / doc_name / "tables" / f"{doc_name}_tables.json"
    ontology_file = project_root / "data" / "ontology" / "DeviceDimension_demo.rdf"

    if not stage2_file.exists():
        logger.error(f"Stage 2 result not found: {stage2_file}")
        sys.exit(1)

    with open(stage2_file, 'r') as f:
        stage2_data = json.load(f)

    if tables_file.exists() and ontology_file.exists():
        extractor = Stage3TableExtractor(
            api_key=api_key,
            ontology_path=str(ontology_file),
        )
        extractor.execute(
            stage2_data=stage2_data,
            tables_path=str(tables_file),
            output_dir=str(base_dir),
            doc_name=doc_name,
        )
    else:
        logger.error("Required files not found")
        logger.error(f"  Tables:   {tables_file} (exists: {tables_file.exists()})")
        logger.error(f"  Ontology: {ontology_file} (exists: {ontology_file.exists()})")
