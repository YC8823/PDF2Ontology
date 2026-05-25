import os
import sys
import json
import logging
import time
import io
import requests
from typing import Dict, Any, List, Tuple
from PIL import Image

try:
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
except ImportError:
    print("Import Error: langchain-core not found. Please install: pip install langchain-core")
    sys.exit(1)

# PyMuPDF
try:
    import fitz 
except ImportError:
    print("Import Error: PyMuPDF not found. Please install: pip install PyMuPDF")
    sys.exit(1)

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
    from src.preprocessors.image_preprocessor import (
        preprocess_data,
        get_images_for_cropping,
        crop_images
    )
except ImportError as e:
    # Fallback/Mock for demonstration if actual files are missing during dev
    print(f"Warning: Preprocessor modules not found ({e}). Ensure src/preprocessors/ exists.")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [EndToEndChain] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DolphinRemoteAnalyzer:
    """
    LangChain Tool/Runnable: Handles interaction with Remote Dolphin Service (RunPod).
    Input: PDF Path
    Output: Path to the saved raw_analysis.json
    """
    def __init__(self, api_url: str = "http://localhost:8080/analyze", batch_size: int = 1):
        self.api_url = api_url
        self.batch_size = batch_size

    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Tuple[Image.Image, int]]:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((img, page_num + 1))
        doc.close()
        return images

    def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the remote analysis logic.
        Expects inputs['pdf_path'] and inputs['output_root'].
        Returns updated inputs with 'raw_json_path'.
        """
        pdf_path = inputs["pdf_path"]
        output_root = inputs["output_root"]
        doc_name = inputs["doc_name"]
        
        logger.info(f">>> STEP 0: Remote Analysis for {doc_name}...")
        
        # Define output path
        raw_json_dir = os.path.join(output_root, "raw_analysis")
        os.makedirs(raw_json_dir, exist_ok=True)
        raw_json_path = os.path.join(raw_json_dir, f"{doc_name}_raw_analysis.json")

        # Check if already exists to save time (Optional caching logic)
        if os.path.exists(raw_json_path):
            logger.info(f"    Found existing raw analysis: {raw_json_path}")
            inputs["raw_json_path"] = raw_json_path
            return inputs

        # 1. Convert
        logger.info("    Converting PDF to images...")
        images = self.convert_pdf_to_images(pdf_path)

        # 2. Send to RunPod
        logger.info(f"    Sending {len(images)} pages to RunPod ({self.api_url})...")
        analysis_results = []
        
        # Simple batch processing
        for i, (img, page_num) in enumerate(images):
            try:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                files = {'file': (f'page_{page_num}.png', img_byte_arr, 'image/png')}
                response = requests.post(self.api_url, files=files, timeout=300)
                response.raise_for_status()
                
                res_json = response.json()
                
                # Normalize result structure
                if isinstance(res_json, dict) and 'results_per_page' in res_json:
                     # Flatten if server returns list inside dict
                    page_data = res_json['results_per_page'][0]
                    page_data['page_number'] = page_num
                    analysis_results.append(page_data)
                else:
                    res_json['page_number'] = page_num
                    analysis_results.append(res_json)
                    
                logger.info(f"    Page {page_num} analyzed.")
            except Exception as e:
                logger.error(f"    Failed to analyze page {page_num}: {e}")
        
        # 3. Save JSON
        final_data = {
            'pdf_filename': os.path.basename(pdf_path),
            'total_pages': len(images),
            'results_per_page': analysis_results
        }
        
        with open(raw_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"    Raw analysis saved: {raw_json_path}")
        
        # Update state
        inputs["raw_json_path"] = raw_json_path
        return inputs


class EndToEndDocumentChain:
    """
    End-to-End Document Processing Chain using LangChain LCEL.
    Flow: PDF -> Dolphin(RunPod) -> JSON -> [Text, Table, Image Extraction]
    """
    
    def __init__(self, output_root: str, api_url: str = "http://localhost:8080/analyze"):
        self.output_root = output_root
        self.dolphin_analyzer = DolphinRemoteAnalyzer(api_url=api_url)
        
    def _initialize_state(self, pdf_path: str) -> Dict[str, Any]:
        """Initializes the state dictionary for the chain"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        return {
            "pdf_path": pdf_path,
            "doc_name": doc_name,
            "output_root": os.path.join(self.output_root, doc_name),
            "raw_json_path": None # Will be filled by Step 0
        }

    def _step_extract_text(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Wrapper for TextExtractor"""
        logger.info(">>> STEP 1: Text Extraction")
        output_path = os.path.join(inputs["output_root"], "text", f"{inputs['doc_name']}_text.json")
        try:
            extractor = TextExtractor(inputs["raw_json_path"])
            extractor.save(output_path)
            return {"status": "success", "file": output_path}
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {"status": "error", "error": str(e)}

    def _step_extract_table(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Wrapper for TableExtractor"""
        logger.info(">>> STEP 2: Table Extraction")
        output_path = os.path.join(inputs["output_root"], "tables", f"{inputs['doc_name']}_tables.json")
        try:
            extractor = TableExtractor(inputs["raw_json_path"], inputs["pdf_path"])
            extractor.extract_and_save(output_path)
            return {"status": "success", "file": output_path}
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return {"status": "error", "error": str(e)}

    def _step_extract_images(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Wrapper for Image Processing"""
        logger.info(">>> STEP 3: Image Processing")
        output_dir = os.path.join(inputs["output_root"], "images")
        output_json = os.path.join(output_dir, f"{inputs['doc_name']}_images.json")
        
        try:
            # Load raw data
            with open(inputs["raw_json_path"], 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Use functional API from image_preprocessor
            _, _, images, tables, paragraphs = preprocess_data(raw_data)
            
            if not images:
                return {"status": "skipped", "reason": "no_images"}
            
            # Matching Logic (Simplified from original class)
            final_archives = []
            for i, img in enumerate(images):
                archive = {
                    "image_id": f"page_{img['page_number']}_img_{i+1}",
                    "page_number": img['page_number'],
                    "image_bbox": img['box'],
                    "caption_info": {},
                    "table_match_info": {}
                }
                # (Assuming find_caption_for_image etc. are available and stateless)
                # ... Add your detailed matching logic here if needed ...
                final_archives.append(archive)

            # Cropping
            images_map = get_images_for_cropping(inputs["pdf_path"])
            crop_images(final_archives, images_map, output_dir, inputs["doc_name"])
            
            # Save
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(final_archives, f, indent=2, ensure_ascii=False)
                
            return {"status": "success", "file": output_json}
        except Exception as e:
            logger.error(f"Image processing failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def build(self):
        """
        Constructs the LangChain Runnable sequence.
        """
        # 1. State Initialization (PDF Path -> State Dict)
        setup_step = RunnableLambda(self._initialize_state)
        
        # 2. Remote Analysis (State Dict -> Updated State Dict with JSON Path)
        dolphin_step = RunnableLambda(self.dolphin_analyzer.analyze)
        
        # 3. Parallel Extraction (State Dict -> Extraction Results)
        # Using RunnableParallel to run text, table, and image extraction independently
        extraction_step = RunnableParallel({
            "text_result": RunnableLambda(self._step_extract_text),
            "table_result": RunnableLambda(self._step_extract_table),
            "image_result": RunnableLambda(self._step_extract_images)
        })
        
        # Compose the Chain using LCEL
        # chain = setup | dolphin | extraction
        chain = setup_step | dolphin_step | extraction_step
        
        return chain

    def run(self, pdf_path: str):
        """Entry point to run the chain"""
        chain = self.build()
        logger.info(f"Starting Chain for: {pdf_path}")
        result = chain.invoke(pdf_path)
        logger.info("Chain execution complete.")
        return result


# ==============================================================================
# Command Line Entry Point
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-End Document Processing Chain (LangChain Powered)")
    parser.add_argument("--pdf", required=True, help="Path to original PDF")
    parser.add_argument("--output", default="data/preprocessing_results", help="Output directory root")
    parser.add_argument("--api_url", default="http://localhost:8080/analyze", help="RunPod API URL")
    
    args = parser.parse_args()
    
    # Instantiate and Run
    pipeline = EndToEndDocumentChain(output_root=args.output, api_url=args.api_url)
    
    try:
        final_results = pipeline.run(args.pdf)
        print("\n=== Final Pipeline Results ===")
        print(json.dumps(final_results, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

