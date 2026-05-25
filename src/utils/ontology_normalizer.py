import sys
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

# ==========================================
# CONFIGURATION
# ==========================================

# 1. Input and Output file paths
input_file = "data/ontology/DeviceDimension_v3.rdf"  # Replace with your actual file path
output_file = "data/ontology/DeviceDimension_v5.rdf"

# 2. The Base URI you want to standardize on.
# IMPORTANT: This should be the common prefix of your ontology WITHOUT the separator.
# Example: If your URIs look like 'http://www.example.org/myonto/ClassA' 
# or 'http://www.example.org/myonto#ClassA', set this to 'http://www.example.org/myonto'
base_uri_string = "http://www.semanticweb.org/yanha/ontologies/2025/7/untitled-ontology-26" 

# ==========================================
# SCRIPT logic
# ==========================================

def normalize_ontology(input_path, output_path, base_uri):
    """
    Reads an RDF file and normalizes all URIs starting with the base_uri
    to use the '#' separator style.
    """
    print(f"Loading ontology from {input_path}...")
    g = Graph()
    try:
        g.parse(input_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Graph loaded with {len(g)} triples.")
    
    # Create a mapping dictionary to store old_uri -> new_uri
    uri_map = {}
    
    # We define the standard base with a hash for the target
    target_base_prefix = base_uri if not base_uri.endswith("#") else base_uri[:-1]
    target_base_prefix_slash = target_base_prefix + "/"
    target_base_prefix_hash = target_base_prefix + "#"

    # Collect all unique URIs in the graph (Subjects, Predicates, Objects)
    all_terms = set()
    for s, p, o in g:
        if isinstance(s, URIRef): all_terms.add(s)
        if isinstance(p, URIRef): all_terms.add(p)
        if isinstance(o, URIRef): all_terms.add(o)

    count_modified = 0

    print("Analyzing URIs...")
    
    for term in all_terms:
        term_str = str(term)
        
        # Check if the term belongs to our ontology namespace
        # We look for terms starting with the base URI (either slash or hash version)
        if term_str.startswith(target_base_prefix):
            
            # Extract the local name (the part after the last / or #)
            # This logic handles both .../Entity and ...#Entity
            if "#" in term_str:
                local_name = term_str.split("#")[-1]
            elif "/" in term_str:
                local_name = term_str.split("/")[-1]
            else:
                # If neither is present after the base (edge case), skip
                continue
            
            # Construct the new standard URI using Hash (#)
            new_uri_str = f"{target_base_prefix}#{local_name}"
            
            # If the URI is effectively different (e.g. it was a slash URI), record it
            if new_uri_str != term_str:
                uri_map[term] = URIRef(new_uri_str)
                count_modified += 1

    if count_modified == 0:
        print("No inconsistent URIs found. The ontology is already clean or the base_uri configuration is incorrect.")
        return

    print(f"Found {count_modified} URIs to normalize. Applying changes...")

    # Create a new graph for the normalized data
    new_g = Graph()
    
    # Preserve namespaces from the original graph
    for prefix, namespace in g.namespaces():
        # If the namespace is the one we are changing (slash version), update it to hash
        if str(namespace).startswith(target_base_prefix):
             new_g.bind(prefix, URIRef(target_base_prefix_hash))
        else:
             new_g.bind(prefix, namespace)
             
    # Ensure the main prefix points to the hash version
    new_g.bind("base", URIRef(target_base_prefix_hash), override=True)

    # Migrate triples
    for s, p, o in g:
        # Normalize Subject
        new_s = uri_map.get(s, s)
        
        # Normalize Predicate
        new_p = uri_map.get(p, p)
        
        # Normalize Object (only if it's a URI)
        new_o = uri_map.get(o, o)
        
        new_g.add((new_s, new_p, new_o))

    print(f"Saving normalized ontology to {output_path}...")
    
    # Serialize to RDF/XML (default for Protege)
    # You can change format to 'turtle' if preferred
    new_g.serialize(destination=output_path, format="xml") 
    print("Done! Open the new file in Protege to verify.")

if __name__ == "__main__":
    normalize_ontology(input_file, output_file, base_uri_string)