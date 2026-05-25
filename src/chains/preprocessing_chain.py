import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

# ==============================================================================
# Module Import Adapter
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.preprocessors.text_preprocessor import TextExtractor
    from src.preprocessors.table_preprocessor import TableExtractor
    from src.preprocessors.image_preprocessor import process_images_with_context
except ImportError as e:
    print(f"Import Error: Please ensure three preprocessor scripts exist in 'src/preprocessors/'.\nDetails: {e}")
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PreprocessingChain] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PreprocessingChain:
    """
    Document Preprocessing Control Chain (Context-Aware Version)
    
    Sequential Execution:
    1. Text Extraction
    2. Table Extraction (with configurable context)
    3. Image Extraction (with configurable context)
    """
    
    def __init__(
        self, 
        raw_json_path: str, 
        pdf_path: str, 
        output_root: str,
        table_context_before: int = 3,
        table_context_after: int = 3,
        image_context_before: int = 3,
        image_context_after: int = 3
    ):
        self.raw_json_path = raw_json_path
        self.pdf_path = pdf_path
        self.output_root = output_root
        self.doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Context window configurations
        self.table_context_before = table_context_before
        self.table_context_after = table_context_after
        self.image_context_before = image_context_before
        self.image_context_after = image_context_after
        
        # Validate files
        if not os.path.exists(raw_json_path):
            raise FileNotFoundError(f"JSON file not found: {raw_json_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Result cache paths
        self.paths = {
            "text": os.path.join(output_root, "text", f"{self.doc_name}_text.json"),
            "table": os.path.join(output_root, "tables", f"{self.doc_name}_tables.json"),
            "image_dir": os.path.join(output_root, "images"),
            "image_json": os.path.join(output_root, "images", f"{self.doc_name}_images.json")
        }
        
        logger.info(f"Preprocessing Chain initialized for: {self.doc_name}")
        logger.info(f"  Table context: before={table_context_before}, after={table_context_after}")
        logger.info(f"  Image context: before={image_context_before}, after={image_context_after}")

    def run_text_extraction(self) -> Dict:
        """Step 1: Extract Plain Text"""
        logger.info(">>> STEP 1: Starting Text Extraction...")
        try:
            extractor = TextExtractor(self.raw_json_path)
            extractor.save(self.paths["text"])
            logger.info(f"    ✓ Text extraction complete: {self.paths['text']}")
            return {"status": "success", "path": self.paths["text"]}
        except Exception as e:
            logger.error(f"    ✗ Text extraction failed: {e}")
            return {"status": "error", "error": str(e)}

    def run_table_extraction(self) -> Dict:
        """Step 2: Extract Table Structures with Context"""
        logger.info(">>> STEP 2: Starting Table Extraction with Context...")
        try:
            extractor = TableExtractor(
                self.raw_json_path, 
                self.pdf_path,
                context_window_before=self.table_context_before,
                context_window_after=self.table_context_after
            )
            extractor.extract_and_save(self.paths["table"])
            logger.info(f"    ✓ Table extraction complete: {self.paths['table']}")
            return {"status": "success", "path": self.paths["table"]}
        except Exception as e:
            logger.error(f"    ✗ Table extraction failed: {e}")
            return {"status": "error", "error": str(e)}

    def run_image_workflow(self) -> Dict:
        """Step 3: Extract Images with Context"""
        logger.info(">>> STEP 3: Starting Image Processing with Context...")
        try:
            # Load raw data
            with open(self.raw_json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Process images with context
            final_archives = process_images_with_context(
                raw_data,
                self.pdf_path,
                self.paths["image_dir"],
                self.doc_name,
                context_window_before=self.image_context_before,
                context_window_after=self.image_context_after
            )
            
            # Save with metadata
            output_path = self.paths["image_json"]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "extraction_config": {
                        "context_window_before": self.image_context_before,
                        "context_window_after": self.image_context_after
                    },
                    "total_images": len(final_archives),
                    "images": final_archives
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"    ✓ Image processing complete: {output_path}")
            return {"status": "success", "count": len(final_archives), "path": output_path}

        except Exception as e:
            logger.error(f"    ✗ Image processing failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def execute(self):
        """Execute the full workflow"""
        start_time = time.time()
        logger.info(f"{'='*70}")
        logger.info(f"Starting Document Processing: {self.doc_name}")
        logger.info(f"{'='*70}")
        
        results = {
            "document": self.doc_name,
            "config": {
                "table_context_before": self.table_context_before,
                "table_context_after": self.table_context_after,
                "image_context_before": self.image_context_before,
                "image_context_after": self.image_context_after
            },
            "steps": {}
        }
        
        # Execute steps
        results["steps"]["text"] = self.run_text_extraction()
        results["steps"]["table"] = self.run_table_extraction()
        results["steps"]["image"] = self.run_image_workflow()
        
        duration = time.time() - start_time
        logger.info(f"{'='*70}")
        logger.info(f"Processing Finished (Duration: {duration:.2f}s)")
        logger.info(f"{'='*70}")
        
        return results


# ==============================================================================
# Command Line Entry Point
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Preprocessing Workflow (Context-Aware)")
    parser.add_argument("--json", default= "data/outputs/runpod_output/EH_01_raw_analysis.json" , help="Path to raw_analysis.json")
    parser.add_argument("--pdf", default= "data/test_materials/EH_01.pdf" , help="Path to original PDF")
    parser.add_argument("--output", default= "data/test_intermediate_results/EH_01", help="Output directory root")
    parser.add_argument("--table_context_before", type=int, default=3, help="Table context window before (default: 3)")
    parser.add_argument("--table_context_after", type=int, default=3, help="Table context window after (default: 3)")
    parser.add_argument("--image_context_before", type=int, default=3, help="Image context window before (default: 3)")
    parser.add_argument("--image_context_after", type=int, default=3, help="Image context window after (default: 3)")
    
    args = parser.parse_args()
    
    chain = PreprocessingChain(
        args.json, 
        args.pdf, 
        args.output,
        table_context_before=args.table_context_before,
        table_context_after=args.table_context_after,
        image_context_before=args.image_context_before,
        image_context_after=args.image_context_after
    )
    chain.execute()
