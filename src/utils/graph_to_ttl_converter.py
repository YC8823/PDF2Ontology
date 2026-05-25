# src/knowledge_extractor/graph_to_ttl_converter.py

import json
import logging
import urllib.parse
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
# Added VOID for dataset description to fix DC.Dataset error
from rdflib.namespace import DC, DCTERMS, VOID

# 尝试导入用户提供的 OntologyLoader
try:
    from src.preprocessors.ontology_loader import OntologyLoaderOwlready2, OntologyLOOS
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.preprocessors.ontology_loader import OntologyLoaderOwlready2, OntologyLOOS

logger = logging.getLogger(__name__)

# 定义常用的命名空间
PROV = Namespace("http://www.w3.org/ns/prov#")

class GraphToTTLConverter:
    """
    将 JSON Graph 转换为 RDF/Turtle (.ttl)。
    
    改进点 (Ref: simple_ttl_merger.py):
    1. 原生 Base IRI 提取：直接分析 TBox 文件文本，获取最准确的 Namespace，防止 rdflib/owlready 自动归一化导致的错位。
    2. 严格属性对齐：优先使用 TBox 中已存在的完整 IRI。
    3. 显式声明：对新属性强制声明类型。
    """

    def __init__(self, ontology_path: str, base_data_uri: str = "http://example.org/data/"):
        self.ontology_path = ontology_path
        self.base_data_uri = base_data_uri
        
        self.graph = Graph()
        self.loos: Optional[OntologyLOOS] = None
        
        # 查找表
        self.property_ranges: Dict[str, URIRef] = {}  # 属性名 -> XSD Type URI
        self.property_uri_map: Dict[str, URIRef] = {} # 属性名 -> 真实 IRI (从 TBox 读取)
        self.declared_properties: Set[URIRef] = set() # 记录已声明类型的属性
        
        # 初始化
        self._initialize_context()
        self._bind_namespaces()

    def _extract_base_iri_from_text(self) -> Optional[str]:
        """
        [借鉴 simple_ttl_merger.py]
        直接从文件文本中提取 Base IRI，作为最可信的来源。
        支持 Turtle (@prefix : <...>) 和 RDF/XML (xml:base="...")
        """
        try:
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                content = f.read(5000) # 只读前5000字符通常够了

            # 1. 尝试匹配 Turtle 格式: @prefix : <http://...>
            match_ttl = re.search(r'@prefix\s+:\s+<([^>]+)>', content)
            if match_ttl:
                logger.info(f"Regex found Turtle Base IRI: {match_ttl.group(1)}")
                return match_ttl.group(1)
            
            # 2. 尝试匹配 Turtle Base: @base <http://...>
            match_base = re.search(r'@base\s+<([^>]+)>', content)
            if match_base:
                logger.info(f"Regex found Turtle @base: {match_base.group(1)}")
                return match_base.group(1)

            # 3. 尝试匹配 RDF/XML: xml:base="http://..."
            match_xml = re.search(r'xml:base=["\']([^"\']+)["\']', content)
            if match_xml:
                logger.info(f"Regex found XML Base IRI: {match_xml.group(1)}")
                return match_xml.group(1)
                
        except Exception as e:
            logger.warning(f"Failed to extract base IRI from text: {e}")
        
        return None

    def _initialize_context(self):
        """加载 Ontology 并解析属性的 Range 和 IRI 定义"""
        logger.info(f"Loading TBox Context from {self.ontology_path}...")
        
        # 1. 首先尝试原生提取 Base IRI
        raw_base_iri = self._extract_base_iri_from_text()
        
        # 2. 使用 Loader 解析结构
        loader = OntologyLoaderOwlready2(self.ontology_path)
        if loader.load() and loader.extract_structure():
            self.loos = loader.loos
            
            # 决策：如果正则提取到了，以此为准；否则信赖 Loader
            if raw_base_iri:
                self.tbox_ns_str = raw_base_iri
            else:
                self.tbox_ns_str = self.loos.base_namespace
                
            self._build_property_maps()
        else:
            logger.warning("Failed to load TBox structure! Fallback mode.")
            self.tbox_ns_str = raw_base_iri if raw_base_iri else "http://example.org/ontology#"
            self.loos = None

        # 确保 Namespace 格式正确 (以 # 或 / 结尾)
        # 注意：如果原文里没有 #/，且是 XML Base，通常需要加 #。
        # 但如果用户明确提供了 simple_ttl_merger 的逻辑，我们尽量保持原样，
        # 除非它明显不能作为 prefix 连接符。
        if not self.tbox_ns_str.endswith(("#", "/")):
            self.tbox_ns_str += "#"
            logger.info(f"Appended '#' to Base IRI: {self.tbox_ns_str}")

    def _build_property_maps(self):
        """构建属性查找表"""
        if not self.loos:
            return

        type_mapping = {
            "int": XSD.integer, "integer": XSD.integer,
            "float": XSD.decimal, "decimal": XSD.decimal, "double": XSD.double,
            "string": XSD.string, "str": XSD.string,
            "bool": XSD.boolean, "boolean": XSD.boolean,
            "date": XSD.date, "datetime": XSD.dateTime
        }

        # --- Datatype Properties ---
        for prop_name, prop_info in self.loos.datatype_properties.items():
            # 优先使用真实 IRI
            if prop_info.iri:
                self.property_uri_map[prop_name] = URIRef(prop_info.iri)
            
            # 映射 Range
            ranges = prop_info.range
            if ranges:
                raw_type = str(ranges[0]).lower()
                mapped_type = None
                for key, xsd_uri in type_mapping.items():
                    if key in raw_type:
                        mapped_type = xsd_uri
                        break
                if mapped_type:
                    self.property_ranges[prop_name] = mapped_type

        # --- Object Properties ---
        for prop_name, prop_info in self.loos.object_properties.items():
            if prop_info.iri:
                self.property_uri_map[prop_name] = URIRef(prop_info.iri)

    def _bind_namespaces(self):
        """绑定前缀"""
        self.TBOX_NS = Namespace(self.tbox_ns_str)
        self.DATA_NS = Namespace(self.base_data_uri)

        # 绑定空前缀通常给 TBox，这样生成的 TTL 最像原来的
        # 但为了避免混淆，这里 explicit bind
        self.graph.bind("tbox", self.TBOX_NS) 
        self.graph.bind("data", self.DATA_NS)
        self.graph.bind("prov", PROV)
        self.graph.bind("owl", OWL)
        self.graph.bind("void", VOID)
        self.graph.bind("dcterms", DCTERMS)

    def _sanitize_id(self, text: str) -> str:
        if not text: return "unknown"
        return urllib.parse.quote(text.replace(" ", "_"))

    def _resolve_property_uri(self, prop_name: str, property_type: str = "datatype") -> URIRef:
        """
        获取属性 URI。
        关键修正：确保使用的是 TBOX_NS，而不是 accidental new namespace。
        """
        # 1. 查表 (最安全)
        if prop_name in self.property_uri_map:
            return self.property_uri_map[prop_name]
        
        # 2. 如果没找到，必须使用当前的 TBOX_NS 构造
        # 这样 Protégé 加载时，如果 Base IRI 对了，就能自动归并
        safe_name = urllib.parse.quote(prop_name.replace(" ", "_"))
        uri = self.TBOX_NS[safe_name]
        
        # 3. 对于不在 TBox 中的新属性，显式声明其类型
        # 否则 Protégé 会将其视为 Annotation Property
        if uri not in self.declared_properties:
            if property_type == "datatype":
                self.graph.add((uri, RDF.type, OWL.DatatypeProperty))
            else:
                self.graph.add((uri, RDF.type, OWL.ObjectProperty))
            
            self.graph.add((uri, RDFS.label, Literal(prop_name)))
            self.declared_properties.add(uri)
            logger.info(f"Declared new property: {uri} as {property_type}")
            
        return uri

    def _create_strict_literal(self, value: Any, prop_name: str) -> Optional[Literal]:
        """严格模式类型转换"""
        target_type = self.property_ranges.get(prop_name)

        if target_type:
            try:
                if target_type == XSD.integer:
                    if isinstance(value, float):
                        if not value.is_integer():
                             raise ValueError(f"Float {value} is not an integer")
                        val_to_cast = int(value)
                    else:
                        val_to_cast = int(value)
                elif target_type in (XSD.decimal, XSD.double, XSD.float):
                    val_to_cast = float(value)
                else:
                    val_to_cast = value

                return Literal(val_to_cast, datatype=target_type)

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"[STRICT MODE] Dropping attribute '{prop_name}': value '{value}' "
                    f"incompatible with TBox range {target_type}."
                )
                return None
        
        return Literal(value)

    def convert(self, json_data: Dict[str, Any], output_path: str):
        instances = json_data.get("instances", [])
        relationships = json_data.get("relationships", [])
        
        logger.info(f"Converting {len(instances)} instances and {len(relationships)} relationships...")
        skipped_count = 0

        # --- Instances ---
        for node in instances:
            node_id = self._sanitize_id(node["id"])
            node_uri = self.DATA_NS[node_id]

            # Type Definition
            class_name = node.get("ontology_class", "Thing")
            # 假设 Class Name 在 TBox 中存在，使用 TBOX_NS
            class_uri = self.TBOX_NS[class_name]
            
            self.graph.add((node_uri, RDF.type, class_uri))
            
            # Label
            label = node.get("label", node_id)
            self.graph.add((node_uri, RDFS.label, Literal(label)))

            # New Class Handling (Dynamic TBox Extension)
            if node.get("is_new_class"):
                # 显式声明这是一个类
                self.graph.add((class_uri, RDF.type, OWL.Class))
                parent = node.get("suggested_parent")
                if parent:
                    parent_uri = self.TBOX_NS[parent]
                    self.graph.add((class_uri, RDFS.subClassOf, parent_uri))
                self.graph.add((class_uri, RDFS.comment, Literal("Auto-discovered class", lang="en")))

            # Attributes (Datatype Properties)
            for attr in node.get("attributes", []):
                p_name = attr.get("property_name")
                val = attr.get("value")
                
                if p_name and val is not None:
                    p_uri = self._resolve_property_uri(p_name, property_type="datatype")
                    
                    lit = self._create_strict_literal(val, p_name)
                    if lit is not None:
                        self.graph.add((node_uri, p_uri, lit))
                    else:
                        skipped_count += 1
            
            # Provenance
            if node.get("cloned_from"):
                source_uri = self.DATA_NS[self._sanitize_id(node["cloned_from"])]
                self.graph.add((node_uri, PROV.wasDerivedFrom, source_uri))

        # --- Relationships (Object Properties) ---
        for rel in relationships:
            subj_id = rel.get("subject_id")
            obj_id = rel.get("object_id")
            pred_name = rel.get("predicate")

            if subj_id and obj_id and pred_name:
                subj_uri = self.DATA_NS[self._sanitize_id(subj_id)]
                obj_uri = self.DATA_NS[self._sanitize_id(obj_id)]
                
                pred_uri = self._resolve_property_uri(pred_name, property_type="object")
                
                self.graph.add((subj_uri, pred_uri, obj_uri))

        # --- Metadata ---
        meta = json_data.get("_metadata", {})
        if meta:
            dataset_node = self.DATA_NS["Dataset"]
            self.graph.add((dataset_node, RDF.type, VOID.Dataset))
            self.graph.add((dataset_node, DCTERMS.created, Literal(meta.get("stage", "final"))))
            if "description" in meta:
                self.graph.add((dataset_node, RDFS.comment, Literal(meta["description"])))

        if skipped_count > 0:
            logger.warning(f"Total attributes dropped by strict validation: {skipped_count}")

        logger.info(f"Serializing RDF to {output_path}...")
        self.graph.serialize(destination=output_path, format="turtle")
        logger.info("Done.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base_dir = Path("data/test_intermediate_results/SS_03")
    json_path = base_dir / "SS_03_stage3_full_graph.json"
    ontology_path = Path("data/ontology/DeviceDimension_v3.rdf")
    output_ttl = base_dir / "SS_03_knowledge_graph.ttl"

    if json_path.exists() and ontology_path.exists():
        converter = GraphToTTLConverter(str(ontology_path), "http://example.org/data/SS_03/")
        with open(json_path, 'r', encoding='utf-8') as f:
            converter.convert(json.load(f), str(output_ttl))