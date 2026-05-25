import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# LangChain LCEL
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableConfig

# =============================================================================
# IMPORT YOUR MODULES
# =============================================================================
# Add project root to path to ensure imports work
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 1. Identity Registry
from src.utils.global_identity_registry import GlobalIdentityRegistry

# 2. Extractors (Adjust paths to match your folder structure)
# Assuming these are inside src/knowledge_extractor based on your descriptions

from src.knowledge_extractor.layered_stage1_text_extractor_refactored import extract_stage1_from_datasheet
from src.knowledge_extractor.layered_stage2_visual_extractor_refactored import execute_stage2_visual_extraction
from src.knowledge_extractor.layered_stage3_table_extractor_refactored import Stage3TableExtractor


# 3. RDF Converter & Loader
from src.knowledge_generator.json_to_rdf_converter import JSONToRDFConverter, OntologyLoader

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Orchestrator] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline"""
    model_name: str = "gpt-4o"
    context_window: int = 3
    base_uri: Optional[str] = None  # None enables auto-detection from TBox

class RawToOntologyChain:
    """
    LCEL-based Orchestrator that connects:
    Raw Inputs -> Preprocessing (simulated) -> Stage 1 -> Stage 2 -> Stage 3 -> RDF Converter
    """

    def __init__(self, tbox_path: str, output_root: str, api_key: str, config: PipelineConfig):
        self.tbox_path = Path(tbox_path)
        self.output_root = Path(output_root)
        self.api_key = api_key
        self.config = config
        
        # Ensure output directory exists
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize Registry
        self.registry = GlobalIdentityRegistry()
        logger.info("Initialized Global Identity Registry")

    def _step_preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 0: Prepare paths and validate inputs.
        In a real scenario, this splits the 'raw_analysis.json' into text/images/tables.
        For now, we assume the paths to these components are provided or inferred.
        """
        raw_path = Path(inputs["raw_json_path"])
        pdf_path = Path(inputs["pdf_path"])
        doc_name = pdf_path.stem
        
        # Define/Expect intermediate file paths
        # (Assuming the pre-processing has already split the raw JSON into these)
        # If your preprocessing is dynamic, call your PreprocessingChain here.
        base_dir = self.output_root / doc_name
        base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"--- Step 0: Setup for {doc_name} ---")
        
        return {
            "doc_name": doc_name,
            "base_dir": base_dir,
            "pdf_path": str(pdf_path),
            "ontology_path": str(self.tbox_path),
            "registry": self.registry,
            
            # Assuming these files exist from Dolphin/Preprocessing output
            # You might need to adjust logic to CREATE them if they don't exist
            "images_json_path": str(base_dir / "images" / f"{doc_name}_images.json"),
            "tables_json_path": str(base_dir / "tables" / f"{doc_name}_tables.json")
        }

    def _step_stage1(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Text Extraction & Skeleton Generation"""
        logger.info("--- Step 1: Text Extraction (Skeleton) ---")
        
        try:
            # Call the functional interface of Stage 1
            result_dict = extract_stage1_from_datasheet(
                pdf_path=state["pdf_path"],
                ontology_path=state["ontology_path"],
                api_key=self.api_key,
                project_root=str(state["base_dir"].parent.parent), # Adjust relative root if needed
                output_dir=str(state["base_dir"]),
                registry=state["registry"]  # Pass the registry!
            )
            
            # The extractor saves the file, but we construct the expected path
            stage1_file = state["base_dir"] / f"{state['doc_name']}_stage1_skeleton.json"
            
            # Verify file creation
            if not stage1_file.exists():
                # If the function returns dict but doesn't save to exact path, save it manually
                with open(stage1_file, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2)
            
            return {**state, "stage1_file": str(stage1_file)}
            
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            raise

    def _step_stage2(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Visual Patching"""
        logger.info("--- Step 2: Visual Patching ---")
        
        images_path = Path(state["images_json_path"])
        if not images_path.exists():
            logger.warning(f"No images file found at {images_path}. Skipping Stage 2 visual enrichment.")
            # Pass Stage 1 output as Stage 2 output to keep chain alive
            return {**state, "stage2_file": state["stage1_file"]}

        try:
            # Call the function from Stage 2 refactored script
            # Note: The execute_stage2... function usually expects file paths
            stage2_output_path = execute_stage2_visual_extraction(
                stage1_result_path=state["stage1_file"],
                images_result_path=state["images_json_path"],
                ontology_path=state["ontology_path"],
                base_dir=str(state["base_dir"]),
                api_key=self.api_key,
                registry=state["registry"] # Pass registry!
            )
            
            return {**state, "stage2_file": stage2_output_path}
            
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            raise

    def _step_stage3(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Table Fission & Instantiation"""
        logger.info("--- Step 3: Table Fission ---")
        
        tables_path = Path(state["tables_json_path"])
        if not tables_path.exists():
            logger.warning(f"No tables file found at {tables_path}. Skipping Stage 3 fission.")
            return {**state, "stage3_file": state["stage2_file"]}

        try:
            # Instantiate Stage 3 class
            extractor = Stage3TableExtractor(
                api_key=self.api_key,
                ontology_path=state["ontology_path"],
                identity_registry=state["registry"]
            )
            
            stage3_output_path = extractor.execute(
                stage2_file=state["stage2_file"],
                tables_file=state["tables_json_path"],
                output_dir=str(state["base_dir"])
            )
            
            return {**state, "stage3_file": stage3_output_path}
            
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            raise

    def _step_convert_to_rdf(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Final Step: JSON to RDF Conversion with TBox Sanitization"""
        logger.info("--- Final Step: RDF Conversion & Merging ---")
        
        json_path = Path(state["stage3_file"])
        output_ttl_path = state["base_dir"] / f"{state['doc_name']}_KnowledgeGraph.ttl"
        
        try:
            # 1. Load the JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                instance_data = json.load(f)
            
            # 2. Initialize Ontology Loader
            logger.info(f"Loading TBox from: {state['ontology_path']}")
            loader = OntologyLoader(state["ontology_path"])
            
            # 3. Initialize Converter
            # Setting base_uri=None enables the Auto-Discovery & Sanitization features
            converter = JSONToRDFConverter(
                ontology_loader=loader,
                base_uri=self.config.base_uri # Default is None
            )
            
            # 4. Convert and Merge
            # merge_tbox=True ensures we get a standalone file with cleaned schema
            graph = converter.convert(instance_data, merge_tbox=True)
            
            # 5. Serialize
            graph.serialize(destination=str(output_ttl_path), format='turtle')
            logger.info(f"Successfully generated Ontology at: {output_ttl_path}")
            
            return {**state, "final_ontology_path": str(output_ttl_path)}

        except Exception as e:
            logger.error(f"RDF Conversion failed: {e}")
            raise

    def get_chain(self):
        """Builds the LCEL Runnable"""
        
        return (
            RunnableLambda(self._step_preprocess)
            | RunnableLambda(self._step_stage1)
            | RunnableLambda(self._step_stage2)
            | RunnableLambda(self._step_stage3)
            | RunnableLambda(self._step_convert_to_rdf)
        )

    def run(self, raw_json_path: str, pdf_path: str):
        """Entry point to execute the pipeline"""
        chain = self.get_chain()
        return chain.invoke({
            "raw_json_path": raw_json_path,
            "pdf_path": pdf_path
        })


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run the full Layered Extraction -> Ontology pipeline")
    parser.add_argument('--raw', required=True, help="Path to raw analysis JSON (or dummy path if files exist)")
    parser.add_argument('--pdf', required=True, help="Path to original PDF")
    parser.add_argument('--tbox', required=True, help="Path to Ontology TBox (RDF/XML)")
    parser.add_argument('--output', required=True, help="Root directory for outputs")
    
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
        
    # Configure Pipeline
    config = PipelineConfig(
        base_uri=None # Let the converter detect it
    )
    
    pipeline = RawToOntologyChain(
        tbox_path=args.tbox,
        output_root=args.output,
        api_key=api_key,
        config=config
    )
    
    try:
        result = pipeline.run(raw_json_path=args.raw, pdf_path=args.pdf)
        print("\n" + "="*80)
        print(f"PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Final Ontology: {result['final_ontology_path']}")
        print("="*80)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)