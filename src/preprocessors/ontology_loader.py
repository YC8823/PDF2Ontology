# src/utils/ontology_loader.py

import logging
import os
from typing import Optional, Dict, List, Any, Union, Set, Tuple
from dataclasses import dataclass, field
import yaml

from owlready2 import *

logger = logging.getLogger(__name__)


@dataclass
class ClassInfo:
    """Represents an OWL Class with comprehensive metadata."""
    python_name: str
    iri: str
    label: Optional[str] = None
    comment: Optional[str] = None
    parent_classes: List[str] = field(default_factory=list)
    subclasses: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)


@dataclass
class PropertyInfo:
    """Represents an OWL Property with domain and range."""
    python_name: str
    iri: str
    property_type: str  # "object", "datatype"
    label: Optional[str] = None
    comment: Optional[str] = None
    domain: List[str] = field(default_factory=list)
    range: List[str] = field(default_factory=list)
    is_functional: bool = False
    inverse_of: Optional[str] = None  # NEW: Inverse property name


@dataclass
class OntologyLOOS:
    """LLM-Optimized Ontology Summary structure."""
    base_namespace: str
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    object_properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    datatype_properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    inverse_property_pairs: List[Tuple[str, str]] = field(default_factory=list)  # NEW


class OntologyLoaderOwlready2:
    """
    Advanced ontology loader using owlready2 for comprehensive TBox extraction.
    """
    
    def __init__(self, ontology_path: str):
        """
        Initialize the loader with path to ontology file.
        
        Args:
            ontology_path: Path to the ontology file (.owl, .rdf, .ttl, etc.)
        """
        self.ontology_path = ontology_path
        self.onto: Optional[Ontology] = None
        self.loos: Optional[OntologyLOOS] = None
        
    def load(self) -> bool:
        """
        Load the ontology from file.
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.ontology_path):
            logger.error(f"Ontology file not found: {self.ontology_path}")
            return False
        
        try:
            self.onto = get_ontology(f"file://{self.ontology_path}").load()
            logger.info(f"Successfully loaded ontology from '{self.ontology_path}'")
            logger.info(f"Ontology IRI: {self.onto.base_iri}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ontology file: {e}", exc_info=True)
            return False
    
    def extract_structure(self) -> Optional[OntologyLOOS]:
        """
        Extract complete ontology structure following LOOS principles.
        
        Returns:
            OntologyLOOS object containing all extracted information
        """
        if not self.onto:
            logger.error("No ontology loaded. Call load() first.")
            return None
        
        base_namespace = self.onto.base_iri
        logger.info(f"Using base namespace: {base_namespace}")
        
        loos = OntologyLOOS(base_namespace=base_namespace)
        
        # Extract classes with metadata and restrictions
        loos.classes = self._extract_classes()
        logger.info(f"Extracted {len(loos.classes)} classes")
        
        # Build class hierarchy
        self._build_class_hierarchy(loos.classes)
        
        # Extract properties with domain and range
        loos.object_properties = self._extract_object_properties()
        loos.datatype_properties = self._extract_datatype_properties()
        
        logger.info(f"Extracted {len(loos.object_properties)} object properties")
        logger.info(f"Extracted {len(loos.datatype_properties)} datatype properties")
        
        # NEW: Extract inverse property relationships
        loos.inverse_property_pairs = self._extract_inverse_properties(loos.object_properties)
        logger.info(f"Extracted {len(loos.inverse_property_pairs)} inverse property pairs")
        
        # Map properties to classes
        self._map_properties_to_classes(loos)
        
        self.loos = loos
        return loos
    
    def _extract_classes(self) -> Dict[str, ClassInfo]:
        """
        Extract all OWL Classes with labels, comments, and restrictions.
        """
        classes = {}
        
        for cls in self.onto.classes():
            python_name = cls.name
            iri = cls.iri
            
            # Get label (first available)
            label = cls.label[0] if cls.label else None
            
            # Get comment (first available)
            comment = cls.comment[0] if cls.comment else None
            
            # Get parent classes and restrictions from is_a
            parent_classes = []
            restrictions = []
            
            for ancestor in cls.is_a:
                if isinstance(ancestor, ThingClass):
                    # Named class - direct inheritance
                    parent_classes.append(ancestor.name)
                elif isinstance(ancestor, Restriction):
                    # Restriction object - logical constraint
                    restriction_text = self._parse_restriction(ancestor)
                    if restriction_text:
                        restrictions.append(restriction_text)
            
            classes[python_name] = ClassInfo(
                python_name=python_name,
                iri=iri,
                label=label,
                comment=comment,
                parent_classes=parent_classes,
                restrictions=restrictions
            )
        
        return classes
    
    def _parse_restriction(self, restriction: Restriction) -> Optional[str]:
        """
        Parse a Restriction object into human-readable text.
        
        Handles:
        - SOME (existential quantification)
        - ONLY (universal quantification)
        - MIN/MAX (cardinality constraints)
        - VALUE (hasValue)
        """
        try:
            prop = restriction.property
            prop_name = prop.name if prop else "unknown"
            
            # Determine restriction type
            if restriction.type == SOME:
                value = restriction.value
                value_name = value.name if hasattr(value, 'name') else str(value)
                return f"{prop_name} some {value_name}"
            
            elif restriction.type == ONLY:
                value = restriction.value
                value_name = value.name if hasattr(value, 'name') else str(value)
                return f"{prop_name} only {value_name}"
            
            elif restriction.type == MIN:
                cardinality = restriction.cardinality
                value = restriction.value
                value_name = value.name if hasattr(value, 'name') else str(value)
                return f"{prop_name} min {cardinality} {value_name}"
            
            elif restriction.type == MAX:
                cardinality = restriction.cardinality
                value = restriction.value
                value_name = value.name if hasattr(value, 'name') else str(value)
                return f"{prop_name} max {cardinality} {value_name}"
            
            elif restriction.type == EXACTLY:
                cardinality = restriction.cardinality
                value = restriction.value
                value_name = value.name if hasattr(value, 'name') else str(value)
                return f"{prop_name} exactly {cardinality} {value_name}"
            
            elif restriction.type == VALUE:
                value = restriction.value
                return f"{prop_name} value {value}"
            
            else:
                return f"{prop_name} (unknown restriction type)"
                
        except Exception as e:
            logger.warning(f"Failed to parse restriction: {e}")
            return None
    
    def _build_class_hierarchy(self, classes: Dict[str, ClassInfo]):
        """Build bidirectional parent-child relationships."""
        for class_name, class_info in classes.items():
            for parent_name in class_info.parent_classes:
                if parent_name in classes:
                    classes[parent_name].subclasses.append(class_name)
    
    def _extract_object_properties(self) -> Dict[str, PropertyInfo]:
        """Extract all OWL Object Properties with domain and range."""
        properties = {}
        
        for prop in self.onto.object_properties():
            python_name = prop.name
            iri = prop.iri
            
            label = prop.label[0] if prop.label else None
            comment = prop.comment[0] if prop.comment else None
            
            # Extract domain (list of classes)
            domain = [cls.name for cls in prop.domain if hasattr(cls, 'name')]
            
            # Extract range (list of classes)
            range_list = [cls.name for cls in prop.range if hasattr(cls, 'name')]
            
            # Check if functional
            is_functional = isinstance(prop, FunctionalProperty)
            
            # NEW: Extract inverse property
            inverse_of = None
            if hasattr(prop, 'inverse_property') and prop.inverse_property:
                inverse_of = prop.inverse_property.name
            
            properties[python_name] = PropertyInfo(
                python_name=python_name,
                iri=iri,
                property_type="object",
                label=label,
                comment=comment,
                domain=domain,
                range=range_list,
                is_functional=is_functional,
                inverse_of=inverse_of
            )
        
        return properties
    
    def _extract_datatype_properties(self) -> Dict[str, PropertyInfo]:
        """Extract all OWL Datatype Properties with domain and range."""
        properties = {}
        
        for prop in self.onto.data_properties():
            python_name = prop.name
            iri = prop.iri
            
            label = prop.label[0] if prop.label else None
            comment = prop.comment[0] if prop.comment else None
            
            # Extract domain (list of classes)
            domain = [cls.name for cls in prop.domain if hasattr(cls, 'name')]
            
            # Extract range (list of data types)
            range_list = []
            for r in prop.range:
                if hasattr(r, '__name__'):
                    range_list.append(r.__name__)
                else:
                    range_list.append(str(r))
            
            # Check if functional
            is_functional = isinstance(prop, FunctionalProperty)
            
            properties[python_name] = PropertyInfo(
                python_name=python_name,
                iri=iri,
                property_type="datatype",
                label=label,
                comment=comment,
                domain=domain,
                range=range_list,
                is_functional=is_functional,
                inverse_of=None  # Datatype properties don't have inverses
            )
        
        return properties
    
    def _extract_inverse_properties(
        self, 
        object_properties: Dict[str, PropertyInfo]
    ) -> List[Tuple[str, str]]:
        """
        Extract inverse property pairs.
        
        Args:
            object_properties: Dictionary of object properties
            
        Returns:
            List of (property1, property2) tuples representing inverse pairs
        """
        inverse_pairs = []
        processed_properties = set()
        
        for prop_name, prop_info in object_properties.items():
            # Skip if already processed
            if prop_name in processed_properties:
                continue
            
            # Check if this property has an inverse
            inverse_name = prop_info.inverse_of
            if inverse_name and inverse_name in object_properties:
                # Create ordered pair (alphabetically for consistency)
                pair = tuple(sorted([prop_name, inverse_name]))
                
                # Avoid duplicates
                if pair not in inverse_pairs:
                    inverse_pairs.append(pair)
                    logger.debug(f"Found inverse pair: {pair[0]} <-> {pair[1]}")
                
                # Mark both as processed
                processed_properties.add(prop_name)
                processed_properties.add(inverse_name)
        
        return inverse_pairs
    
    def _map_properties_to_classes(self, loos: OntologyLOOS):
        """Map properties to their domain classes."""
        for prop_dict in [loos.object_properties, loos.datatype_properties]:
            for prop_name, prop_info in prop_dict.items():
                for domain_class in prop_info.domain:
                    if domain_class in loos.classes:
                        loos.classes[domain_class].properties.append(prop_name)
    
    def export_to_yaml(self) -> str:
        """
        Export to YAML format following LOOS principles.
        
        Returns:
            YAML-formatted string optimized for LLM consumption
        """
        if not self.loos:
            logger.error("No structure extracted. Call extract_structure() first.")
            return ""
        
        # Build LOOS structure
        loos_dict = {
            "ontology": {
                "namespace": self.loos.base_namespace,
                "classes": {}
            }
        }
        
        # Add classes with all metadata
        for class_name, class_info in self.loos.classes.items():
            class_dict = {
                "label": class_info.label if class_info.label else class_name,
                "description": class_info.comment if class_info.comment else "",
            }
            
            if class_info.parent_classes:
                class_dict["inherits_from"] = class_info.parent_classes
            
            if class_info.properties:
                class_dict["properties"] = class_info.properties
            
            if class_info.restrictions:
                class_dict["constraints"] = class_info.restrictions
            
            loos_dict["ontology"]["classes"][class_name] = class_dict
        
        # Add object properties
        if self.loos.object_properties:
            loos_dict["ontology"]["object_properties"] = {}
            for prop_name, prop_info in self.loos.object_properties.items():
                prop_dict = {
                    "label": prop_info.label if prop_info.label else prop_name,
                    "description": prop_info.comment if prop_info.comment else "",
                }
                
                if prop_info.domain:
                    prop_dict["domain"] = prop_info.domain
                
                if prop_info.range:
                    prop_dict["range"] = prop_info.range
                
                if prop_info.is_functional:
                    prop_dict["functional"] = True
                
                # NEW: Add inverse property information
                if prop_info.inverse_of:
                    prop_dict["inverse_of"] = prop_info.inverse_of
                
                loos_dict["ontology"]["object_properties"][prop_name] = prop_dict
        
        # NEW: Add inverse property pairs section
        if self.loos.inverse_property_pairs:
            loos_dict["ontology"]["inverse_property_pairs"] = [
                {"property1": pair[0], "property2": pair[1]}
                for pair in self.loos.inverse_property_pairs
            ]
        
        # Add datatype properties
        if self.loos.datatype_properties:
            loos_dict["ontology"]["datatype_properties"] = {}
            for prop_name, prop_info in self.loos.datatype_properties.items():
                prop_dict = {
                    "label": prop_info.label if prop_info.label else prop_name,
                    "description": prop_info.comment if prop_info.comment else "",
                }
                
                if prop_info.domain:
                    prop_dict["domain"] = prop_info.domain
                
                if prop_info.range:
                    prop_dict["range"] = prop_info.range
                
                if prop_info.is_functional:
                    prop_dict["functional"] = True
                
                loos_dict["ontology"]["datatype_properties"][prop_name] = prop_dict
        
        return yaml.dump(loos_dict, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export complete structure as dictionary.
        
        Returns:
            Dictionary containing all ontology information
        """
        if not self.loos:
            return {}
        
        return {
            "base_namespace": self.loos.base_namespace,
            "classes": {
                name: {
                    "python_name": info.python_name,
                    "iri": info.iri,
                    "label": info.label,
                    "comment": info.comment,
                    "parent_classes": info.parent_classes,
                    "subclasses": info.subclasses,
                    "properties": info.properties,
                    "restrictions": info.restrictions
                }
                for name, info in self.loos.classes.items()
            },
            "object_properties": {
                name: {
                    "python_name": info.python_name,
                    "iri": info.iri,
                    "label": info.label,
                    "comment": info.comment,
                    "domain": info.domain,
                    "range": info.range,
                    "is_functional": info.is_functional,
                    "inverse_of": info.inverse_of
                }
                for name, info in self.loos.object_properties.items()
            },
            "datatype_properties": {
                name: {
                    "python_name": info.python_name,
                    "iri": info.iri,
                    "label": info.label,
                    "comment": info.comment,
                    "domain": info.domain,
                    "range": info.range,
                    "is_functional": info.is_functional
                }
                for name, info in self.loos.datatype_properties.items()
            },
            "inverse_property_pairs": [
                {"property1": pair[0], "property2": pair[1]}
                for pair in self.loos.inverse_property_pairs
            ]
        }


# Convenience function
def load_and_serialize_to_yaml(ontology_path: str) -> Optional[str]:
    """
    Load ontology and return YAML-formatted LOOS for LLM consumption.
    
    Args:
        ontology_path: Path to ontology file (.owl, .rdf, .ttl, etc.)
        
    Returns:
        YAML-formatted string, or None if failed
    """
    loader = OntologyLoaderOwlready2(ontology_path)
    
    if not loader.load():
        return None
    
    if not loader.extract_structure():
        return None
    
    return loader.export_to_yaml()


# Example usage
if __name__ == "__main__":
    import sys
    import json
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example path - update as needed
    ontology_path = "data/ontology/DeviceDimension_v3.rdf"
    
    if not os.path.exists(ontology_path):
        print(f"Ontology file not found: {ontology_path}")
        print("Please update the path to point to your ontology file (.owl, .rdf, .ttl, etc.).")
        sys.exit(1)
    
    print(f"Loading ontology from: {ontology_path}\n")
    
    # Create loader
    loader = OntologyLoaderOwlready2(ontology_path)
    
    # Load and extract
    if not loader.load():
        print("Failed to load ontology")
        sys.exit(1)
    
    loos = loader.extract_structure()
    if not loos:
        print("Failed to extract structure")
        sys.exit(1)
    
    # Print summary
    print("="*60)
    print("ONTOLOGY SUMMARY")
    print("="*60)
    print(f"Base Namespace: {loos.base_namespace}")
    print(f"Classes: {len(loos.classes)}")
    print(f"Object Properties: {len(loos.object_properties)}")
    print(f"Datatype Properties: {len(loos.datatype_properties)}")
    print(f"Inverse Property Pairs: {len(loos.inverse_property_pairs)}")
    
    # Print inverse property pairs
    if loos.inverse_property_pairs:
        print("\nInverse Property Pairs:")
        for pair in loos.inverse_property_pairs:
            print(f"  • {pair[0]} <-> {pair[1]}")
    print()
    
    # Export to YAML
    print("="*60)
    print("LOOS YAML EXPORT")
    print("="*60)
    yaml_output = loader.export_to_yaml()
    print(yaml_output)
    
    # Save outputs
    output_dir = "data/ontology"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save YAML
    yaml_path = os.path.join(output_dir, "ontology_loos_v3.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_output)
    print(f"YAML exported to: {yaml_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, "ontology_structure_v3.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(loader.export_to_dict(), f, indent=2, ensure_ascii=False)
    print(f"JSON exported to: {json_path}")