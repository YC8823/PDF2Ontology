# src/utils/merge_and_validate_ttl.py
"""
TTL Merge and Validation Tool

Merges extracted ABox triples with TBox ontology and validates using reasoners.

Features:
1. Robust merging using RDFLib (Graph Union)
2. Custom Java path support for Protege users
3. Smart Namespace separator detection (/ vs #)
"""

import os
import sys
import logging
import json
import shutil
import subprocess
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import datetime 

# FIX: Import rdflib to handle reliable merging
try:
    from owlready2 import *
    import rdflib
    from rdflib.namespace import RDF, OWL, RDFS
except ImportError as e:
    print(f"Error: Required library not installed ({e})")
    print("Please install required packages:")
    print("pip install owlready2 rdflib")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTLMergeValidator:
    """
    Merges extracted ABox with TBox and validates using reasoners.
    """
    
    def __init__(
        self,
        tbox_path: str,
        abox_path: str,
        java_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        self.tbox_path = tbox_path
        self.abox_path = abox_path
        self.java_path = java_path
        self.metadata_path = metadata_path
        
        # Validate inputs
        if not os.path.exists(tbox_path):
            raise FileNotFoundError(f"TBox file not found: {tbox_path}")
        if not os.path.exists(abox_path):
            raise FileNotFoundError(f"ABox file not found: {abox_path}")
        
        # Load metadata if available
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded extraction metadata from {metadata_path}")
        
        self.merged_onto = None
        self.validation_results = {}
        self.base_iri = None 

    def _find_java(self) -> str:
        """
        Find java executable. Prioritizes user argument, then system path, then common paths.
        """
        # 1. User provided path
        if self.java_path:
            if os.path.exists(self.java_path):
                logger.info(f"Using provided Java path: {self.java_path}")
                return self.java_path
            else:
                logger.warning(f"Provided Java path does not exist: {self.java_path}")
        
        # 2. Check system path
        try:
            subprocess.run(["java", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return "java"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        # 3. Check common Windows Protege paths
        common_paths = [
            r"C:\Program Files\Protege-5.5.0\jre\bin\java.exe",
            r"C:\Program Files\Protege-5.2.0\jre\bin\java.exe",
            r"C:\Program Files (x86)\Protege-5.5.0\jre\bin\java.exe",
        ]
        
        # Search for any java.exe in Program Files if specific versions fail
        if os.name == 'nt':
            for root_dir in [r"C:\Program Files", r"C:\Program Files (x86)"]:
                if os.path.exists(root_dir):
                    for folder in os.listdir(root_dir):
                        if "Protege" in folder or "Java" in folder or "jdk" in folder or "jre" in folder:
                            java_path = os.path.join(root_dir, folder, "bin", "java.exe")
                            # Check deeper for Protege's jre structure
                            java_path_protege = os.path.join(root_dir, folder, "jre", "bin", "java.exe")
                            
                            if os.path.exists(java_path):
                                logger.info(f"Found Java at: {java_path}")
                                return java_path
                            if os.path.exists(java_path_protege):
                                logger.info(f"Found Java at: {java_path_protege}")
                                return java_path_protege

        return None

    def _get_base_iri_rdflib(self) -> str:
        """
        Extract Base IRI robustly using RDFLib by analyzing the Ontology declaration 
        AND existing classes to determine the correct separator (/ or #).
        """
        g = rdflib.Graph()
        try:
            g.parse(self.tbox_path)
        except:
            try:
                g.parse(self.tbox_path, format='turtle')
            except:
                pass
                
        base = None
        # 1. Look for <subject> a owl:Ontology
        for s, p, o in g.triples((None, RDF.type, OWL.Ontology)):
            base = str(s)
            break
            
        if not base:
             logger.warning("Could not find owl:Ontology declaration. Using default.")
             return "http://example.org/ontology#"

        # 2. Smart Separator Detection
        # Check if base already ends with a separator
        if base.endswith('#') or base.endswith('/'):
            logger.info(f"  TBox defines explicit Base IRI: {base}")
            return base

        # 3. If no separator, scan existing classes to see what they use
        separator = '#' # Default fallback
        
        # Look for any subject that starts with the base IRI
        for s, p, o in g.triples((None, RDF.type, OWL.Class)):
            s_str = str(s)
            if s_str.startswith(base):
                # Check what comes immediately after the base
                suffix = s_str[len(base):]
                if suffix.startswith('/'):
                    separator = '/'
                    logger.info("  Detected '/' separator from existing classes.")
                    break
                elif suffix.startswith('#'):
                    separator = '#'
                    logger.info("  Detected '#' separator from existing classes.")
                    break
        
        final_base = base + separator
        return final_base

    def _preprocess_abox(self, abox_content: str) -> str:
        """
        Preprocess ABox triples to ensure proper namespace.
        """
        logger.info("Preprocessing ABox triples...")
        
        # Get the exact Base IRI from TBox
        self.base_iri = self._get_base_iri_rdflib()
        logger.info(f"  Using Base IRI: {self.base_iri}")
        
        # Build header
        ttl_header = f"""# Merged ABox with TBox
# Generated: {datetime.datetime.now().isoformat()} 
# TBox source: {os.path.basename(self.tbox_path)}
# ABox source: {os.path.basename(self.abox_path)}

@prefix : <{self.base_iri}> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"""
        # Clean content
        lines = abox_content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('@prefix') or stripped.startswith('#'):
                continue
            if 'owl:imports' in stripped:
                continue
            cleaned_lines.append(line)
        
        cleaned_abox = '\n'.join(cleaned_lines).strip()
        logger.info(f"✓ Preprocessed ABox ({len(cleaned_lines)} lines)")
        return ttl_header + '\n' + cleaned_abox
    
    def _merge_abox_into_tbox(self, output_path: str) -> bool:
        """
        Merge using RDFLib Graph Union.
        """
        logger.info("\n" + "="*70)
        logger.info("Merging ABox with TBox (RDFLib Engine)...")
        logger.info("="*70)
        
        try:
            # 1. Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 2. Read and Preprocess ABox
            with open(self.abox_path, 'r', encoding='utf-8') as f:
                abox_content = f.read()
            preprocessed_abox = self._preprocess_abox(abox_content)

            # 3. Parse TBox
            logger.info(f"  Parsing TBox: {self.tbox_path}")
            graph = rdflib.Graph()
            # Try parsing TBox (auto-detect format)
            try:
                graph.parse(self.tbox_path)
            except Exception as e:
                # Fallback for common TBox formats
                if self.tbox_path.endswith('.ttl'):
                    graph.parse(self.tbox_path, format='turtle')
                else:
                    graph.parse(self.tbox_path, format='xml')
            
            initial_len = len(graph)
            logger.info(f"  TBox triples: {initial_len}")

            # 4. Parse ABox into the SAME graph (Merge)
            logger.info("  Parsing and Merging ABox...")
            try:
                graph.parse(data=preprocessed_abox, format='turtle')
            except Exception as e:
                logger.error(f"Failed to parse ABox Turtle: {e}")
                raise

            final_len = len(graph)
            new_triples = final_len - initial_len
            logger.info(f"✓ Merge successful! Added {new_triples} new triples.")
            
            # 5. Serialize merged graph to output
            logger.info(f"  Saving merged ontology to: {output_path}")
            graph.serialize(destination=output_path, format='xml') 
            
            # 6. Load the MERGED result into Owlready2 (New World for isolation)
            logger.info("  Reloading merged file into Owlready2...")
            
            # Create a new isolated world to ensure clean loading
            self.new_world = World()
            self.merged_onto = self.new_world.get_ontology(f"file://{output_path}").load()
            
            # Recalculate stats
            classes = len(list(self.merged_onto.classes()))
            individuals = len(list(self.merged_onto.individuals()))
            
            logger.info("\nMerge Statistics (Owlready2 View):")
            logger.info(f"  Total classes: {classes}")
            logger.info(f"  Total individuals: {individuals}")
            
            if individuals == 0 and new_triples > 0:
                 logger.warning(f"! Owlready2 still sees 0 individuals. Detected Base IRI was: {self.base_iri}")
                 logger.warning("  Please check if this matches your TBox exactly (including http vs https and # vs /).")
            
            return True
            
        except Exception as e:
            logger.error(f"Merge failed: {e}", exc_info=True)
            return False

    def _validate_with_reasoner(self) -> Dict[str, Any]:
        """
        Validate using HermiT (default) or Pellet. Handles Java detection.
        """
        logger.info("\n" + "="*70)
        logger.info("Validating with Reasoner (HermiT)...")
        logger.info("="*70)
        
        results = {
            "consistent": False,
            "inferred_facts": 0,
            "errors": [],
            "warnings": []
        }

        # 1. FIND JAVA
        java_exe = self._find_java()
        if not java_exe:
            msg = "Java runtime not found! Reasoner skipped."
            logger.warning(f"⚠ {msg}")
            logger.warning("  Please provide path to java.exe using --java_path argument.")
            results["warnings"].append(msg)
            results["errors"].append("Java not installed - Validation skipped")
            self.validation_results = results
            return results
        
        # Configure owlready2 to use the found java
        if java_exe != "java":
            owlready2.JAVA_EXE = java_exe
            logger.info(f"  Using Java at: {java_exe}")
        
        try:
            logger.info("Running consistency check...")
            
            # Use the merged ontology object
            with self.merged_onto:
                try:
                    # Sync reasoner (HermiT is default and usually bundled)
                    # We switched from sync_reasoner_pellet to sync_reasoner
                    sync_reasoner(
                        infer_property_values=True,
                        infer_data_property_values=True,
                        debug=0
                    )
                    
                    results["consistent"] = True
                    logger.info("✓ Ontology is CONSISTENT")
                    
                    # Count inferred facts
                    inferred_count = 0
                    for ind in self.merged_onto.individuals():
                        inferred_count += len(ind.INDIRECT_is_a) - len(ind.is_a)
                    
                    results["inferred_facts"] = inferred_count
                    if inferred_count > 0:
                        logger.info(f"✓ Reasoner inferred {inferred_count} additional facts")
                    
                except OwlReadyInconsistentOntologyError as e:
                    results["consistent"] = False
                    results["errors"].append(f"Inconsistency: {str(e)}")
                    logger.error(f"✗ Ontology is INCONSISTENT: {e}")
                except Exception as e:
                    raise e
                
        except Exception as e:
            err_msg = str(e)
            if "WinError 2" in err_msg or "No such file" in err_msg:
                 logger.error("✗ Failed to run Reasoner: Java found but execution failed.")
                 results["errors"].append(f"Reasoner execution failed: {err_msg}")
            else:
                logger.error(f"Validation error: {e}")
                results["errors"].append(f"Reasoner error: {str(e)}")
        
        self.validation_results = results
        return results
    
    def _check_property_usage(self) -> Dict[str, Any]:
        """
        Analyze property usage.
        """
        logger.info("\nAnalyzing property usage...")
        stats = {"object_properties": {}, "data_properties": {}, "unused_properties": []}
        
        if not self.merged_onto:
            return stats

        try:
            # Check object property usage
            for prop in self.merged_onto.object_properties():
                usage_count = 0
                for ind in self.merged_onto.individuals():
                    if getattr(ind, prop.python_name, None):
                        usage_count += 1
                stats["object_properties"][prop.name] = usage_count
                if usage_count == 0:
                    stats["unused_properties"].append(f"Object: {prop.name}")
            
            # Check data property usage
            for prop in self.merged_onto.data_properties():
                usage_count = 0
                for ind in self.merged_onto.individuals():
                    if getattr(ind, prop.python_name, None):
                        usage_count += 1
                stats["data_properties"][prop.name] = usage_count
                if usage_count == 0:
                    stats["unused_properties"].append(f"Data: {prop.name}")
            
            # Log
            used_obj = sum(1 for c in stats["object_properties"].values() if c > 0)
            logger.info(f"  Object properties used: {used_obj}/{len(stats['object_properties'])}")
            
        except Exception as e:
            logger.error(f"Property analysis error: {e}")
        
        return stats
    
    def _generate_validation_report(self, output_path: str, property_stats: Dict[str, Any]) -> bool:
        """Generate Markdown report."""
        logger.info(f"\nGenerating validation report...")
        try:
            report_path = output_path.replace('.owl', '_validation_report.md')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Ontology Merge Report\n\n")
                f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Validation Section
                f.write("## 1. Validation Status\n\n")
                if self.validation_results.get("consistent"):
                    f.write("✅ **CONSISTENT**\n\n")
                    f.write(f"- Inferred facts: {self.validation_results.get('inferred_facts', 0)}\n")
                else:
                    errs = self.validation_results.get("errors", [])
                    if any("Java" in e for e in errs):
                        f.write("⚠ **SKIPPED** (Java not found)\n\n")
                    else:
                        f.write("❌ **INCONSISTENT / ERROR**\n\n")
                    
                    for err in errs:
                        f.write(f"- {err}\n")

                # Statistics
                f.write("\n## 2. Statistics\n\n")
                if self.merged_onto:
                    f.write(f"- **Classes:** {len(list(self.merged_onto.classes()))}\n")
                    f.write(f"- **Individuals:** {len(list(self.merged_onto.individuals()))}\n")
                
                f.write("\n## 3. Unused Properties\n\n")
                if property_stats["unused_properties"]:
                    for prop in property_stats["unused_properties"]:
                        f.write(f"- `{prop}`\n")
                else:
                    f.write("All properties are used.\n")
                    
            logger.info(f"✓ Validation report saved to {report_path}")
            return True
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return False

    def merge_and_validate(self, output_path: str) -> bool:
        logger.info("="*70)
        logger.info("TTL MERGE AND VALIDATION (RDFLib + Owlready2)")
        logger.info("="*70)
        
        # 1. Merge (RDFLib)
        if not self._merge_abox_into_tbox(output_path):
            return False
        
        # 2. Validate (HermiT)
        self._validate_with_reasoner()
        
        # 3. Stats
        property_stats = self._check_property_usage()
        
        # 4. Report
        self._generate_validation_report(output_path, property_stats)
        
        logger.info("\n" + "="*70)
        logger.info("PROCESS COMPLETE")
        logger.info("="*70)
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge and Validate TTL")
    parser.add_argument('--tbox', default= "data/ontology/DeviceDimension_v3.rdf")
    parser.add_argument('--abox', default= "data/outputs/one_shot_extraction/t58740en_extracted_abox.ttl")
    parser.add_argument('--output', default= "data/outputs/merged_ontology/merged_device_dimensions.owl")
    parser.add_argument('--metadata', default=None)
    # Add new argument for Java path
    parser.add_argument('--java_path', default=None, help='C:\Dokument\MA_Cao\Materials\Ontology\Protege\Protege-5.6.5')
    
    args = parser.parse_args()
    
    try:
        merger = TTLMergeValidator(
            tbox_path=args.tbox,
            abox_path=args.abox,
            java_path=args.java_path,
            metadata_path=args.metadata
        )
        merger.merge_and_validate(args.output)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()