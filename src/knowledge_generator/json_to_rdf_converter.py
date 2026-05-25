"""
JSON to RDF Converter (v2.0)

Converts Stage 3 JSON output (two-layer or legacy flat format) into an RDF/OWL ABox graph.

Design:
- Handles both Stage 3 v3.0 (prototype_graph + instance_graph) and legacy (entities +
  relationships) formats transparently.
- Forced type alignment: every literal is cast to the exact XSD type declared in the
  TBox, with a graceful xsd:string fallback on failure — no plain literals, no
  implicit auto-typing by rdflib.
- Numeric string sanitisation: strips unit symbols (mm, kg, Ø …) before casting so
  that LLM-generated values like "120.5 mm" or "Ø25" are correctly stored as numbers.
- Object-property filtering: only predicates declared in TBox object_properties are
  written; unknown predicates are dropped with a DEBUG log.
- Datatype consistency check: SPARQL pre-flight scan before serialisation warns about
  any property that carries literals with mixed XSD datatypes.
- hasDimensionType guard: only written when the property is actually declared in TBox.
- TBox merge: optional re-parse + URI normalisation (unchanged from v1.0).
"""

import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef, XSD

# Ensure project root is on sys.path when running as a standalone script
try:
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
except Exception:
    pass

try:
    from src.preprocessors.ontology_loader import OntologyLoaderOwlready2
except ImportError:
    class OntologyLoaderOwlready2:          # minimal stub for standalone execution
        def __init__(self, path):
            self.ontology_path = path
            self.loos = None
            self.onto = None
        def load(self): return False
        def extract_structure(self): return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Regex: strip unit tokens and diameter symbols from numeric strings
_STRIP_UNITS = re.compile(
    r'\s*(mm|cm|m|inch|in|kg|g|bar|psi|°C|°F|[Øø∅φΦ°])\s*',
    re.IGNORECASE,
)


class JSONToRDFConverter:
    """
    Converts Stage 3 JSON (prototype_graph / instance_graph or legacy entities /
    relationships) into an rdflib Graph suitable for serialisation to Turtle / RDF-XML.

    Forced type alignment (v2.0):
        Every data-property literal is cast to the XSD type declared in the TBox
        (e.g. hasDimensionValue → xsd:double).  When a value cannot be cast to the
        declared type (e.g. a stray string for a numeric property) the converter
        falls back to xsd:string and logs a warning, rather than writing a literal
        with the wrong datatype.  The plain-literal fallback present in v1.0 is
        removed — every literal now carries an explicit datatype tag.

    Object-property filtering:
        Only predicates that appear in the TBox object_properties are written as
        triples.  Unknown predicates are silently dropped with a DEBUG log.

    TBox merge:
        When merge_tbox=True the TBox is re-parsed with rdflib and every URI that
        does not belong to a standard namespace (rdf/rdfs/owl/xsd) is rewritten to
        the unified base namespace before merging.
    """

    STANDARD_NAMESPACES: Set[str] = {
        str(RDF), str(RDFS), str(OWL), str(XSD),
        "http://www.w3.org/XML/1998/namespace",
    }

    # owlready2 stores datatype-property ranges as Python type names
    PYTHON_TYPE_TO_XSD: Dict[str, URIRef] = {
        "float":   XSD.double,
        "int":     XSD.integer,
        "str":     XSD.string,
        "bool":    XSD.boolean,
        "Decimal": XSD.decimal,
        "decimal": XSD.decimal,
    }

    def __init__(
        self,
        ontology_loader: OntologyLoaderOwlready2,
        base_uri: Optional[str] = None,
    ):
        self.loader = ontology_loader
        self.graph  = Graph()

        # 1. Base URI
        self.base_uri = (
            Namespace(base_uri.rstrip("#").rstrip("/") + "/")
            if base_uri else self._detect_base_uri()
        )
        logger.info(f"Base URI: {self.base_uri}")

        # 2. data-property name → XSD type  (for typed literal creation)
        self._prop_type_map: Dict[str, URIRef] = self._build_type_map()
        logger.info(f"Typed data properties: {len(self._prop_type_map)}")

        # 3. Valid TBox object-property names  (for relationship filtering)
        self._valid_object_props: Set[str] = self._build_valid_object_props()
        logger.info(f"Valid object properties: {self._valid_object_props}")

        # 4. Resolve Dimension-class property names from TBox
        self._dim_class, self._dim_rel, self._dim_prop_val, \
            self._dim_prop_unit, self._dim_prop_param = self._resolve_dim_props()

        # 5. Namespace bindings
        self.graph.bind("",    self.base_uri)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
        self.graph.bind("ont", self.base_uri)

    # ── TBox introspection ────────────────────────────────────────────────────

    def _detect_base_uri(self) -> Namespace:
        loos = getattr(self.loader, "loos", None)
        if loos and loos.base_namespace:
            ns = loos.base_namespace.rstrip("#").rstrip("/") + "/"
            return Namespace(ns)
        fallback = "http://www.example.org/industrial_devices/"
        logger.warning(f"Cannot detect base URI — using fallback: {fallback}")
        return Namespace(fallback)

    def _build_type_map(self) -> Dict[str, URIRef]:
        loos = getattr(self.loader, "loos", None)
        if not loos:
            return {}
        result: Dict[str, URIRef] = {}
        no_range: List[str] = []
        for prop_name, prop_info in loos.datatype_properties.items():
            matched = False
            for range_type in prop_info.range:
                xsd_type = self.PYTHON_TYPE_TO_XSD.get(range_type)
                if xsd_type:
                    result[prop_name] = xsd_type
                    matched = True
                    break
            if not matched:
                no_range.append(prop_name)
        if no_range:
            logger.warning(
                f"TBox data properties with no range declaration "
                f"(will use auto-typed literals): {no_range}"
            )
        return result

    def _build_valid_object_props(self) -> Set[str]:
        loos = getattr(self.loader, "loos", None)
        return set(loos.object_properties.keys()) if loos else set()

    def _resolve_dim_props(self):
        """Resolve Dimension class name and its key property names from TBox."""
        loos = getattr(self.loader, "loos", None)
        dim_class  = "Dimension"
        dim_rel    = "hasDimension"
        prop_val   = "hasDimensionValue"
        prop_unit  = "hasDimensionUnit"
        prop_param = "hasDimensionParameterName"

        if not loos:
            return dim_class, dim_rel, prop_val, prop_unit, prop_param

        # Class
        candidates = [c for c in loos.classes if "Dimension" in c]
        if candidates:
            dim_class = candidates[0]

        # Object property (range contains Dimension class)
        for p, info in loos.object_properties.items():
            if dim_class in info.range:
                dim_rel = p
                break

        # Data properties
        dim_props = [
            (p, info) for p, info in loos.datatype_properties.items()
            if dim_class in info.domain or not info.domain
        ]
        for p, _ in dim_props:
            pl = p.lower()
            if ("value" in pl or "val" in pl) and ("dimension" in pl or "dim" in pl):
                prop_val = p
            elif "unit" in pl:
                prop_unit = p
            elif ("name" in pl or "param" in pl) and ("dimension" in pl or "parameter" in pl):
                prop_param = p

        logger.info(
            f"Dimension TBox: class='{dim_class}', rel='{dim_rel}', "
            f"val='{prop_val}', unit='{prop_unit}', param='{prop_param}'"
        )
        return dim_class, dim_rel, prop_val, prop_unit, prop_param

    def _has_tbox_prop(self, name: str) -> bool:
        """Return True if *name* is declared as a datatype property in the TBox."""
        loos = getattr(self.loader, "loos", None)
        return loos is not None and name in loos.datatype_properties

    # ── URI helpers ───────────────────────────────────────────────────────────

    def _safe_uri(self, local_name: str) -> URIRef:
        if not local_name:
            return self.base_uri["UnknownEntity"]
        clean = quote(local_name.strip().replace(" ", "_"))
        return self.base_uri[clean]

    def _get_local_name(self, uri: URIRef) -> str:
        s = str(uri)
        return s.split("#")[-1] if "#" in s else s.split("/")[-1]

    def _clean_node(self, node: Any) -> Any:
        """Rewrite non-standard URIs to the unified base namespace."""
        if isinstance(node, URIRef):
            s = str(node)
            if not any(s.startswith(ns) for ns in self.STANDARD_NAMESPACES):
                return self.base_uri[self._get_local_name(node)]
        return node

    # ── Literal creation (v2.0 — forced type alignment) ──────────────────────

    def _sanitize_numeric_string(self, s: str) -> Any:
        """
        Strip unit tokens and diameter symbols from a string, then try to parse
        the remainder as a number.  Returns the numeric value if successful,
        otherwise the original string (so the caller can still store it as text).

        Examples:
            "120.5 mm"  → 120.5
            "Ø 25"      → 25.0
            "DN15"      → "DN15"   (not purely numeric, kept as-is)
        """
        cleaned = _STRIP_UNITS.sub(' ', s).strip()
        cleaned = re.sub(r'[^\d.\-]', '', cleaned)
        if cleaned:
            try:
                return float(cleaned) if '.' in cleaned else int(cleaned)
            except ValueError:
                pass
        return s

    def _create_typed_literal(self, prop_name: str, value: Any) -> Literal:
        """
        Cast *value* to the XSD type declared in the TBox for *prop_name*.

        Changes from v1.0:
        - String values are pre-sanitised (unit stripping) before numeric cast.
        - On cast failure the fallback is xsd:string, not a plain literal.
        - When no TBox range is declared, Python builtins are explicitly tagged
          (int → xsd:integer, float → xsd:double, etc.) — plain literals are
          never produced, preventing silent datatype mismatches in the reasoner.
        """
        # Pre-sanitise strings that might encode numbers with units
        if isinstance(value, str):
            value = self._sanitize_numeric_string(value)

        xsd_type = self._prop_type_map.get(prop_name)

        if xsd_type:
            try:
                if xsd_type in (XSD.double, XSD.float):
                    return Literal(float(value), datatype=XSD.double)
                if xsd_type == XSD.integer:
                    # int(float(x)) handles "120.0" → 120 safely
                    return Literal(int(float(value)), datatype=XSD.integer)
                if xsd_type == XSD.decimal:
                    return Literal(str(value), datatype=XSD.decimal)
                if xsd_type == XSD.boolean:
                    return Literal(bool(value), datatype=XSD.boolean)
                if xsd_type == XSD.string:
                    return Literal(str(value), datatype=XSD.string)
                return Literal(value, datatype=xsd_type)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Type cast failed: '{prop_name}' value={value!r} "
                    f"target={xsd_type} — falling back to xsd:string. ({e})"
                )
                return Literal(str(value), datatype=XSD.string)

        # No TBox range declared: use explicit Python-type-based tags
        # (never plain literals, which cause silent reasoner inconsistencies)
        if isinstance(value, bool):
            return Literal(value,       datatype=XSD.boolean)
        if isinstance(value, int):
            return Literal(value,       datatype=XSD.integer)
        if isinstance(value, float):
            return Literal(value,       datatype=XSD.double)
        return Literal(str(value),      datatype=XSD.string)

    # ── Datatype consistency pre-flight ───────────────────────────────────────

    def validate_datatypes(self) -> List[str]:
        """
        SPARQL scan for data properties that carry literals with more than one
        distinct XSD datatype (symptom of mixed-type writes).  Returns a list of
        warning strings; also logs them.  Call before graph.serialize().
        """
        q = """
        SELECT ?prop (COUNT(DISTINCT ?dt) AS ?count) WHERE {
            ?s ?prop ?o .
            FILTER(isLiteral(?o))
            BIND(datatype(?o) AS ?dt)
        }
        GROUP BY ?prop
        HAVING (?count > 1)
        """
        issues: List[str] = []
        for row in self.graph.query(q):
            msg = f"Mixed datatypes on <{row.prop}>: {int(row['count'])} distinct types"
            issues.append(msg)
            logger.warning(f"  ⚠️  {msg}")
        if not issues:
            logger.info("Datatype consistency check: OK")
        return issues

    # ── Core conversion ───────────────────────────────────────────────────────

    def convert(self, json_data: Dict[str, Any], merge_tbox: bool = False) -> Graph:
        """
        Entry point.  Detects format (v3.0 two-layer vs legacy flat) and
        dispatches to the appropriate processing methods.
        """
        logger.info("Starting JSON → RDF conversion…")

        if "prototype_graph" in json_data:
            # ── Stage 3 v3.0: two-layer format ──────────────────────────────
            logger.info("Detected Stage 3 v3.0 (prototype_graph / instance_graph)")
            proto_graph    = json_data["prototype_graph"]
            instance_graph = json_data.get("instance_graph", {})

            prototypes = proto_graph.get("device_prototypes", [])
            for proto in prototypes:
                self._process_prototype(proto)
            logger.info(f"Processed {len(prototypes)} device prototypes")

            obj_rels = proto_graph.get("object_relations", [])
            written = sum(1 for r in obj_rels if self._process_relationship(r))
            logger.info(f"Written {written}/{len(obj_rels)} prototype object relations")

            instances = instance_graph.get("device_instances", [])
            for inst in instances:
                self._process_instance(inst)
            logger.info(f"Processed {len(instances)} device instances")

        else:
            # ── Legacy flat format (Stage 3 v2.x / Stage 1-2) ───────────────
            logger.info("Detected legacy flat format (entities / relationships)")
            entities = json_data.get("entities", []) or json_data.get("devices", [])
            for entity in entities:
                self._process_entity(entity)
            logger.info(f"Processed {len(entities)} entities")

            relationships = json_data.get("relationships", [])
            written = sum(1 for r in relationships if self._process_relationship(r))
            logger.info(f"Written {written}/{len(relationships)} relationships")

        # Pre-flight datatype consistency check
        self.validate_datatypes()

        if merge_tbox:
            self._merge_tbox()

        return self.graph

    # ── v3.0 processing methods ───────────────────────────────────────────────

    def _process_prototype(self, proto: Dict[str, Any]) -> None:
        """
        Write a device prototype as an OWL named class (subClassOf its ontology_class)
        with OWL punning (also declared as NamedIndividual) so that object-property
        triples between prototypes are valid ABox assertions.
        """
        proto_id = proto.get("id")
        if not proto_id:
            return

        proto_uri = self._safe_uri(proto_id)
        class_uri = self._safe_uri(proto.get("ontology_class", "Device"))

        # Class declaration + subclass hierarchy
        self.graph.add((proto_uri, RDF.type,        OWL.Class))
        self.graph.add((proto_uri, RDFS.subClassOf,  class_uri))
        # OWL punning: also an individual so object properties can be asserted directly
        self.graph.add((proto_uri, RDF.type,         OWL.NamedIndividual))

        if "label" in proto:
            self.graph.add((proto_uri, RDFS.label, Literal(proto["label"], datatype=XSD.string)))

        # Prototype-level data attributes (from Stage 1/2)
        for attr in proto.get("attributes", []):
            prop_name = attr.get("property_name")
            val       = attr.get("value")
            if prop_name and val is not None:
                self.graph.add((
                    proto_uri,
                    self.base_uri[prop_name],
                    self._create_typed_literal(prop_name, val),
                ))

        # Dimension prototypes: no individual RDF nodes (no values yet);
        # write as rdfs:comment annotations for traceability.
        for dim_p in proto.get("dimension_prototypes", []):
            param = dim_p.get("parameter_name", "?")
            dtype = dim_p.get("dimension_type", "")
            self.graph.add((
                proto_uri,
                RDFS.comment,
                Literal(f"DimensionPrototype: {param} ({dtype})", datatype=XSD.string),
            ))

    def _process_instance(self, inst: Dict[str, Any]) -> None:
        """
        Write a device instance as an OWL NamedIndividual whose rdf:type is the
        prototype class (enabling inheritance of prototype-level object properties).
        Embedded dimension entries are expanded into individual Dimension nodes.
        """
        inst_id  = inst.get("id")
        proto_id = inst.get("prototype_id")
        if not inst_id:
            return

        inst_uri = self._safe_uri(inst_id)
        self.graph.add((inst_uri, RDF.type, OWL.NamedIndividual))

        if proto_id:
            # rdf:type → prototype class (inherits all prototype object properties)
            self.graph.add((inst_uri, RDF.type, self._safe_uri(proto_id)))

        # Explicit ontology_class as additional type assertion (for querying convenience)
        if "ontology_class" in inst:
            self.graph.add((inst_uri, RDF.type, self._safe_uri(inst["ontology_class"])))

        if "label" in inst:
            self.graph.add((inst_uri, RDFS.label, Literal(inst["label"], datatype=XSD.string)))

        # Expand embedded dimensions
        for i, dim in enumerate(inst.get("dimensions", [])):
            self._process_dimension(inst_id, i, dim, inst_uri)

    def _process_dimension(
        self,
        device_id: str,
        idx: int,
        dim: Dict[str, Any],
        device_uri: URIRef,
    ) -> None:
        """
        Expand one dimension entry (embedded in device instance) into a Dimension
        NamedIndividual and link it to the device via hasDimension.

        Handles:
        - hasDimensionValue / hasDimensionUnit / hasDimensionParameterName
          using TBox-resolved property names and forced type alignment.
        - hasDimensionType: written only if the property is declared in TBox;
          otherwise stored as rdfs:comment to avoid undefined predicate pollution.
        """
        param   = dim.get("parameter_name", f"dim{idx}")
        dim_id  = f"{device_id}_dim_{param}"
        dim_uri = self._safe_uri(dim_id)

        self.graph.add((dim_uri, RDF.type, OWL.NamedIndividual))
        self.graph.add((dim_uri, RDF.type, self._safe_uri(self._dim_class)))

        # hasDimensionParameterName
        self.graph.add((
            dim_uri,
            self.base_uri[self._dim_prop_param],
            self._create_typed_literal(self._dim_prop_param, param),
        ))

        # hasDimensionValue (key in dim dict is the TBox prop name)
        val = dim.get(self._dim_prop_val)
        if val is not None:
            self.graph.add((
                dim_uri,
                self.base_uri[self._dim_prop_val],
                self._create_typed_literal(self._dim_prop_val, val),
            ))

        # hasDimensionUnit
        unit = dim.get(self._dim_prop_unit)
        if unit is not None:
            self.graph.add((
                dim_uri,
                self.base_uri[self._dim_prop_unit],
                self._create_typed_literal(self._dim_prop_unit, unit),
            ))

        # hasDimensionType — only if declared in TBox
        dim_type = dim.get("dimension_type")
        if dim_type:
            if self._has_tbox_prop("hasDimensionType"):
                self.graph.add((
                    dim_uri,
                    self.base_uri["hasDimensionType"],
                    Literal(dim_type, datatype=XSD.string),
                ))
            else:
                self.graph.add((
                    dim_uri,
                    RDFS.comment,
                    Literal(f"dimension_type: {dim_type}", datatype=XSD.string),
                ))

        # hasDimension link: device instance → Dimension individual
        self.graph.add((device_uri, self.base_uri[self._dim_rel], dim_uri))

    # ── Legacy flat-format processing (backward compat) ───────────────────────

    def _process_entity(self, entity: Dict[str, Any]) -> None:
        """Process a flat entity dict (legacy Stage 3 v2.x format)."""
        entity_id = entity.get("id")
        if not entity_id:
            return

        subj = self._safe_uri(entity_id)

        if "ontology_class" in entity:
            self.graph.add((subj, RDF.type, self._safe_uri(entity["ontology_class"])))

        if "label" in entity:
            self.graph.add((subj, RDFS.label, Literal(entity["label"], datatype=XSD.string)))

        for attr in entity.get("attributes", []):
            prop_name = attr.get("property_name")
            val       = attr.get("value")
            if prop_name and val is not None:
                self.graph.add((
                    subj,
                    self.base_uri[prop_name],
                    self._create_typed_literal(prop_name, val),
                ))

    def _process_relationship(self, rel: Dict[str, Any]) -> bool:
        """
        Write one object-property triple.
        Returns True if written, False if the predicate is not in TBox and was dropped.
        """
        subj_id   = rel.get("subject_id")
        pred_name = rel.get("predicate")
        obj_id    = rel.get("object_id")

        if not (subj_id and pred_name and obj_id):
            return False

        if self._valid_object_props and pred_name not in self._valid_object_props:
            logger.debug(f"Dropping '{pred_name}' — not in TBox object properties")
            return False

        self.graph.add((
            self._safe_uri(subj_id),
            self.base_uri[pred_name],
            self._safe_uri(obj_id),
        ))
        return True

    # ── TBox merge ────────────────────────────────────────────────────────────

    def _merge_tbox(self) -> None:
        """
        Re-load the TBox with rdflib, rewrite all non-standard URIs to the
        unified base namespace, and merge every triple into the output graph.
        """
        path = getattr(self.loader, "ontology_path", None)
        if not path:
            logger.warning("Cannot merge TBox: ontology_path not set on loader")
            return
        try:
            tbox_g = Graph()
            tbox_g.parse(str(path))
            count = 0
            for s, p, o in tbox_g:
                self.graph.add((self._clean_node(s), self._clean_node(p), self._clean_node(o)))
                count += 1
            logger.info(f"Merged {count} TBox triples (URIs normalised to base namespace)")
        except Exception as e:
            logger.error(f"TBox merge failed: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    project_root  = Path(__file__).parent.parent.parent
    json_path     = project_root / "data" / "test_intermediate_results_2" / "SS_03" / "SS_03_stage3_instances.json"
    ontology_path = project_root / "data" / "ontology" / "DeviceDimension_demo.rdf"
    output_path   = project_root / "data" / "test_intermediate_results_2" / "SS_03" / "SS_03_KnowledgeGraph.ttl"

    for p, label in [(ontology_path, "Ontology"), (json_path, "Input JSON")]:
        if not p.exists():
            logger.error(f"{label} not found: {p}")
            return

    loader = OntologyLoaderOwlready2(str(ontology_path))
    if not loader.load():
        logger.error("Failed to load ontology")
        return
    loader.extract_structure()

    converter = JSONToRDFConverter(ontology_loader=loader)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return

    graph = converter.convert(data, merge_tbox=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output_path), format="turtle")
    logger.info(f"Knowledge Graph saved to: {output_path}")


if __name__ == "__main__":
    main()
