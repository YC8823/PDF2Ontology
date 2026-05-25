"""
Enhanced TTL Merger - Support for RDF/XML input and new class definitions

Improvements:
1. Detect new classes created by LLM in ABox
2. Add correct rdfs:subClassOf declarations for new classes
3. Ensure new classes use correct base IRI
4. Support RDF/XML format TBox input (auto-convert to TTL)
"""

import os
import re
import logging
from rdflib import Graph, RDF, OWL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_rdf_to_ttl(rdf_path: str) -> str:
    """
    Convert RDF/XML file to TTL format string
    
    Args:
        rdf_path: Path to RDF/XML file
    
    Returns:
        TTL format string
    """
    logger.info(f"Converting RDF/XML to TTL format: {rdf_path}")
    
    try:
        graph = Graph()
        graph.parse(rdf_path, format='xml')
        
        ttl_content = graph.serialize(format='turtle')
        
        logger.info(f"✓ Converted RDF to TTL ({len(ttl_content)} characters)")
        return ttl_content
        
    except Exception as e:
        logger.error(f"Failed to convert RDF to TTL: {e}")
        raise


def detect_file_format(file_path: str) -> str:
    """
    Detect file format based on extension and content
    
    Args:
        file_path: Path to ontology file
    
    Returns:
        'ttl' or 'rdf'
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.ttl', '.turtle']:
        return 'ttl'
    elif ext in ['.rdf', '.owl', '.xml']:
        return 'rdf'
    else:
        # Try to detect from content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('<?xml') or first_line.startswith('<rdf:RDF'):
                    return 'rdf'
                elif first_line.startswith('@prefix') or first_line.startswith('@base'):
                    return 'ttl'
        except Exception:
            pass
    
    logger.warning(f"Cannot determine format for {file_path}, assuming TTL")
    return 'ttl'


def extract_base_iri(tbox_content: str) -> str:
    """Extract base IRI from TBox TTL content (regex fallback)"""
    match = re.search(r'@prefix\s+:\s+<([^>]+)>', tbox_content)
    if match:
        return match.group(1)

    match = re.search(r'@base\s+<([^>]+)>', tbox_content)
    if match:
        return match.group(1)

    logger.warning("Could not find base IRI")
    return ""


def extract_base_iri_from_graph(tbox_path: str) -> str:
    """Extract base IRI from TBox file using rdflib graph traversal"""
    graph = Graph()
    graph.parse(tbox_path)

    # Prefer explicit owl:Ontology IRI
    for s in graph.subjects(RDF.type, OWL.Ontology):
        base = str(s)
        if not base.endswith('#'):
            base += '#'
        return base

    # Fallback: infer from any class URI
    for s in graph.subjects(RDF.type, OWL.Class):
        uri = str(s)
        if '#' in uri:
            return uri.rsplit('#', 1)[0] + '#'

    logger.warning("Could not find base IRI")
    return ""


def extract_all_prefixes(tbox_content: str) -> str:
    """Extract all prefix declarations"""
    prefix_lines = []
    for line in tbox_content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('@prefix'):
            prefix_lines.append(line)

    return '\n'.join(prefix_lines)


def extract_tbox_classes(tbox_content: str) -> set:
    """Extract class names from serialized TTL (regex, prefix-dependent)."""
    classes = set()

    pattern1 = r':(\w+)\s+rdf:type\s+owl:Class'
    for match in re.finditer(pattern1, tbox_content):
        classes.add(match.group(1))

    pattern2 = r':(\w+)\s+a\s+owl:Class'
    for match in re.finditer(pattern2, tbox_content):
        classes.add(match.group(1))

    return classes


def extract_tbox_classes_from_graph(tbox_path: str) -> set:
    """
    Extract all OWL class local names from the TBox file using rdflib.
    Works regardless of prefix names in the serialised output.
    """
    graph = Graph()
    graph.parse(tbox_path)

    classes = set()
    for subject in graph.subjects(RDF.type, OWL.Class):
        uri = str(subject)
        local_name = uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
        if local_name:
            classes.add(local_name)

    logger.info(f"Found {len(classes)} classes in TBox")
    logger.debug(f"TBox classes: {sorted(classes)}")
    return classes


def find_new_classes_in_abox(abox_content: str, tbox_classes: set) -> dict:
    """
    Find new classes defined in ABox (inferred from multiple type declarations)
    
    LLM uses multiple type declarations to express hierarchy, e.g.:
    :Type3274Actuator rdf:type :ElectrohydraulicActuatorComponent, :ValveActuator, :DeviceComponent
    
    Here ElectrohydraulicActuatorComponent is a new class, need to infer its parent
    
    Returns:
        Dict: {class_name: parent_class_name}
        e.g.: {'ElectrohydraulicActuatorComponent': 'ValveActuator'}
    """
    new_classes = {}
    
    # Find all instance type declarations
    # Pattern: :Instance rdf:type :Class1, :Class2, :Class3
    instance_pattern = r':(\w+)\s+(?:rdf:type|a)\s+(:[^;\.]+)'
    
    for match in re.finditer(instance_pattern, abox_content):
        instance_name = match.group(1)
        types_string = match.group(2)
        
        # Parse multiple types (comma-separated)
        type_names = []
        for type_match in re.finditer(r':(\w+)', types_string):
            type_name = type_match.group(1)
            type_names.append(type_name)
        
        if not type_names:
            continue
        
        # Check for new classes (not in TBox)
        for type_name in type_names:
            if type_name not in tbox_classes and type_name not in new_classes:
                # This is a new class, need to infer parent
                # Find most specific TBox class from same instance's other types
                tbox_types_for_this_instance = [
                    t for t in type_names if t in tbox_classes
                ]
                
                if tbox_types_for_this_instance:
                    # Find most specific parent (first in list is usually most specific)
                    parent_class = find_most_specific_parent(
                        tbox_types_for_this_instance, 
                        tbox_classes
                    )
                    new_classes[type_name] = parent_class
                    logger.info(
                        f"Found new class: {type_name} (from instance {instance_name})\n"
                        f"  Instance types: {type_names}\n"
                        f"  Inferred parent: {parent_class}"
                    )
                else:
                    # No TBox classes found, infer from class name
                    parent_class = infer_parent_class(type_name, tbox_classes)
                    new_classes[type_name] = parent_class
                    logger.warning(
                        f"New class '{type_name}' has no TBox types in same instance, "
                        f"inferred parent from name: {parent_class}"
                    )
    
    return new_classes


def find_most_specific_parent(candidate_parents: list, tbox_classes: set) -> str:
    """
    Find most specific parent from candidates
    
    Strategy: Assume type list is ordered from most specific to most general
    Return first class in list
    
    Example: [ElectrohydraulicActuatorComponent, ValveActuator, DeviceComponent]
             → Return ValveActuator (first TBox class)
    """
    if not candidate_parents:
        return 'DeviceComponent'  # Default parent
    
    # Return first (assumed most specific)
    return candidate_parents[0]


def infer_parent_class(class_name: str, tbox_classes: set) -> str:
    """
    Infer parent class from class name
    
    Strategy:
    1. If class name contains a known class name, use that as parent
       Example: Insulated3241ValveBody → ValveBody
    2. Otherwise return 'DeviceComponent' as default parent
    """
    # Sort by length (prioritize matching more specific class names)
    sorted_classes = sorted(tbox_classes, key=len, reverse=True)
    
    for tbox_class in sorted_classes:
        if tbox_class in class_name:
            logger.debug(f"Inferred parent: {class_name} → {tbox_class}")
            return tbox_class
    
    # Default parent
    default_parent = 'DeviceComponent'
    logger.debug(f"Using default parent: {class_name} → {default_parent}")
    return default_parent


def ensure_new_class_definitions(
    abox_content: str, 
    new_classes: dict, 
    base_iri: str
) -> str:
    """
    Add complete OWL class definitions for new classes
    
    For each new class:
    1. Add rdf:type owl:Class declaration
    2. Add rdfs:subClassOf declaration (pointing to inferred parent)
    
    Note: Does not modify instance's multiple type declarations, only adds class definitions
    """
    if not new_classes:
        return abox_content
    
    logger.info(f"Adding OWL class definitions for {len(new_classes)} new classes...")
    
    # Build new class definition block
    new_class_definitions = []
    new_class_definitions.append("# New Classes Defined by LLM")
    new_class_definitions.append("# (Inferred from instance type declarations)")
    new_class_definitions.append("")
    
    for class_name, parent_name in sorted(new_classes.items()):
        # Create class definition
        definition = [
            f":{class_name} rdf:type owl:Class ;",
            f"    rdfs:subClassOf :{parent_name} .",
            ""
        ]
        new_class_definitions.extend(definition)
        logger.info(f"Added class definition: {class_name} rdfs:subClassOf {parent_name}")
    
    # Insert new class definitions at start of ABox (after prefixes)
    lines = abox_content.split('\n')
    insert_position = 0
    
    # Find end of prefix block
    for i, line in enumerate(lines):
        if line.strip().startswith('@prefix'):
            insert_position = i + 1
    
    # Skip empty lines after prefixes
    while insert_position < len(lines) and not lines[insert_position].strip():
        insert_position += 1
    
    # Insert new class definitions
    new_class_block = '\n'.join(new_class_definitions) + '\n'
    lines.insert(insert_position, new_class_block)
    
    return '\n'.join(lines)


def add_prefixes_to_abox(abox_content: str, prefix_block: str, base_iri: str) -> str:
    """Add prefix declarations to ABox"""
    
    if '@prefix' in abox_content:
        logger.info("ABox already has prefixes")
        
        if '@prefix :' not in abox_content:
            logger.info("Adding default prefix to ABox")
            default_prefix = f"@prefix : <{base_iri}> .\n"
            abox_content = re.sub(r'(@prefix)', default_prefix + r'\1', abox_content, count=1)
        
        return abox_content
    else:
        logger.info("Adding all prefixes to ABox")
        return prefix_block + '\n\n' + abox_content


def validate_new_classes(new_classes: dict, tbox_classes: set) -> bool:
    """
    Validate that parent classes of new classes exist in TBox
    
    Returns:
        True if all valid, False otherwise
    """
    all_valid = True
    
    for class_name, parent_name in new_classes.items():
        if parent_name not in tbox_classes:
            logger.error(
                f"Invalid new class definition: {class_name} → {parent_name}\n"
                f"  Parent class '{parent_name}' does not exist in TBox!"
            )
            all_valid = False
        else:
            logger.debug(f"✓ Valid: {class_name} → {parent_name}")
    
    return all_valid


def merge_ttl_files(
    tbox_path: str, 
    abox_path: str, 
    output_path: str,
    validate_classes: bool = True
) -> bool:
    """
    Merge TBox and ABox, with support for new class definitions
    
    Args:
        tbox_path: TBox file path (supports .ttl, .rdf, .owl formats)
        abox_path: ABox TTL file path
        output_path: Output file path
        validate_classes: Whether to validate new classes
    
    Returns:
        Success status
    """
    logger.info("="*70)
    logger.info("Enhanced TTL Merger (RDF/TTL Support + New Classes)")
    logger.info("="*70)
    logger.info(f"TBox: {tbox_path}")
    logger.info(f"ABox: {abox_path}")
    logger.info(f"Output: {output_path}")
    
    # Check files
    if not os.path.exists(tbox_path):
        logger.error(f"TBox not found: {tbox_path}")
        return False
    
    if not os.path.exists(abox_path):
        logger.error(f"ABox not found: {abox_path}")
        return False
    
    # Detect and read TBox
    logger.info("\n[1/7] Reading TBox...")
    tbox_format = detect_file_format(tbox_path)
    logger.info(f"  Detected format: {tbox_format.upper()}")
    
    if tbox_format == 'rdf':
        # Convert RDF/XML to TTL
        tbox_content = convert_rdf_to_ttl(tbox_path)
    else:
        # Read TTL directly
        with open(tbox_path, 'r', encoding='utf-8') as f:
            tbox_content = f.read()
    
    logger.info(f"  TBox size: {len(tbox_content)} chars")
    
    # Extract TBox information
    logger.info("\n[2/7] Extracting TBox structure...")
    base_iri = extract_base_iri_from_graph(tbox_path)
    if not base_iri:
        base_iri = extract_base_iri(tbox_content)
    logger.info(f"  Base IRI: {base_iri}")

    prefix_block = extract_all_prefixes(tbox_content)
    num_prefixes = prefix_block.count('@prefix')
    logger.info(f"  Prefixes: {num_prefixes}")

    # Use rdflib-based extraction: robust against any prefix naming in serialised TTL
    tbox_classes = extract_tbox_classes_from_graph(tbox_path)
    logger.info(f"  TBox classes: {len(tbox_classes)}")
    
    # Read ABox
    logger.info("\n[3/7] Reading and analyzing ABox...")
    with open(abox_path, 'r', encoding='utf-8') as f:
        abox_content = f.read()
    logger.info(f"  ABox size: {len(abox_content)} chars")
    
    # Find new classes
    logger.info("\n[4/7] Detecting new classes in ABox...")
    new_classes = find_new_classes_in_abox(abox_content, tbox_classes)
    
    if new_classes:
        logger.info(f"  Found {len(new_classes)} new classes:")
        for class_name, parent_name in sorted(new_classes.items()):
            logger.info(f"    • {class_name} → {parent_name}")
        
        # Validate new classes
        if validate_classes:
            logger.info("\n[5/7] Validating new classes...")
            if not validate_new_classes(new_classes, tbox_classes):
                logger.error("Validation failed! Some parent classes do not exist in TBox.")
                logger.error("Please fix the new class definitions or disable validation.")
                return False
            logger.info("  ✓ All new classes are valid")
        
        # Ensure new classes have complete definitions
        logger.info("\n[6/7] Adding class definitions...")
        abox_content = ensure_new_class_definitions(abox_content, new_classes, base_iri)
    else:
        logger.info("  No new classes detected")
        logger.info("\n[5/7] Validation skipped (no new classes)")
        logger.info("\n[6/7] No class definitions to add")
    
    # Add prefixes
    logger.info("\n[7/7] Merging and saving...")
    abox_content = add_prefixes_to_abox(abox_content, prefix_block, base_iri)
    
    # Merge
    separator = "\n\n" + "#"*70 + "\n"
    separator += "# ABox - Extracted Instances\n"
    separator += "#"*70 + "\n\n"
    
    merged_content = tbox_content + separator + abox_content
    
    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    file_size = os.path.getsize(output_path) / 1024
    logger.info(f"  Saved: {file_size:.1f} KB")
    
    # Statistics
    num_individuals = merged_content.count('rdf:type :')
    logger.info(f"\nStatistics:")
    logger.info(f"  TBox classes: {len(tbox_classes)}")
    logger.info(f"  New classes: {len(new_classes)}")
    logger.info(f"  Total classes: {len(tbox_classes) + len(new_classes)}")
    logger.info(f"  Estimated individuals: {num_individuals}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ MERGE COMPLETED")
    logger.info("="*70)
    logger.info(f"Output: {output_path}")
    logger.info("="*70)
    
    return True


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced TTL Merger - Support for RDF/XML input and LLM-created new classes"
    )
    parser.add_argument('--tbox', default='data/ontology/DeviceDimension_no_anchor.rdf', 
                        help='TBox file (supports .ttl, .rdf, .owl)')
    parser.add_argument('--abox', default='data/test_results/one_shot_extraction/_extracted_abox.ttl', 
                        help='ABox TTL file')
    parser.add_argument('--output', default='data/outputs/one_shot_extraction/merged_ontology/merged_ontology.ttl', 
                        help='Output merged TTL file')
    parser.add_argument('--no-validate', action='store_true', 
                        help='Skip validation of new classes')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = merge_ttl_files(
        tbox_path=args.tbox,
        abox_path=args.abox,
        output_path=args.output,
        validate_classes=not args.no_validate
    )
    
    sys.exit(0 if success else 1)


