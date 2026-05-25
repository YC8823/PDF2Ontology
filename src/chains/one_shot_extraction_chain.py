"""
PDF to TTL Chain - Complete Pipeline using LangChain Expression Language

This chain connects:
1. OneShotTripleExtractor: PDF -> Extracted ABox TTL
2. SimpleTTLMerger: TBox + ABox -> Merged Ontology TTL
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not installed. Token estimation will be unavailable. Install with: pip install tiktoken")

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

# Setup paths
try:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Import components
from src.knowledge_extractor.one_shot_triple_extractor import OneShotTripleExtractor
from src.knowledge_generator.ttl_merger import merge_ttl_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def estimate_tokens_simple(text: str) -> int:
    """
    Estimate token count for text using tiktoken (o200k_base encoding)

    This encoding is used by gpt-5.1 models.

    Args:
        text: Text to count tokens
    
    Returns:
        Estimated token count (0 if tiktoken unavailable)
    """
    if not TIKTOKEN_AVAILABLE:
        # Rough fallback: ~4 characters per token
        return len(text) // 4
    
    try:
        # o200k_base is used by gpt-4o and gpt-5.1
        encoding = tiktoken.get_encoding('o200k_base')
        return len(encoding.encode(text))
    except Exception:
        # Fallback
        return len(text) // 4


def estimate_image_tokens(width: int, height: int, detail: str = "high") -> int:
    """
    Estimate tokens for image using OpenAI's formula
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        detail: "high" or "low"
    
    Returns:
        Estimated token count
    """
    if detail == "low":
        return 85
    
    # High detail calculation (OpenAI's formula)
    # Step 1: Fit into 2048x2048
    scale = min(2048 / width, 2048 / height, 1.0)
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    
    # Step 2: Scale shortest side to 768px
    scale2 = 768 / min(scaled_width, scaled_height)
    final_width = int(scaled_width * scale2)
    final_height = int(scaled_height * scale2)
    
    # Step 3: Count 512x512 tiles
    tiles_x = (final_width + 511) // 512
    tiles_y = (final_height + 511) // 512
    
    # Each tile = 170 tokens, plus 85 base tokens
    return tiles_x * tiles_y * 170 + 85


def estimate_pdf_extraction_tokens(
    num_pages: int,
    dpi: int,
    prompt_length_chars: int
) -> Dict[str, Any]:
    """
    Estimate total tokens for PDF extraction
    
    Uses o200k_base encoding (gpt-4o/gpt-5.1) and standard pricing.
    
    Args:
        num_pages: Number of PDF pages
        dpi: DPI setting
        prompt_length_chars: Approximate prompt text length
    
    Returns:
        Token estimation breakdown
    """
    # Estimate prompt text tokens
    text_tokens = estimate_tokens_simple("x" * prompt_length_chars)
    
    # Estimate image tokens (assume A4 size: 210mm x 297mm)
    width = int(210 * dpi / 25.4)  # mm to pixels
    height = int(297 * dpi / 25.4)
    tokens_per_image = estimate_image_tokens(width, height, detail="high")
    total_image_tokens = tokens_per_image * num_pages
    
    # Total prompt tokens
    prompt_tokens = text_tokens + total_image_tokens
    
    # Estimate completion (conservative: 15k for typical datasheet)
    completion_tokens = 15000
    
    # Calculate cost using gpt-4o pricing as reference
    # (gpt-5.1 pricing similar, adjust multiplier if needed)
    prompt_cost = (prompt_tokens / 1_000_000) * 2.50
    completion_cost = (completion_tokens / 1_000_000) * 10.00
    cost_usd = prompt_cost + completion_cost
    
    return {
        'num_pages': num_pages,
        'dpi': dpi,
        'image_resolution': f"{width}x{height}px",
        'prompt_text_tokens_est': text_tokens,
        'prompt_image_tokens_est': total_image_tokens,
        'tokens_per_image': tokens_per_image,
        'prompt_tokens_est': prompt_tokens,
        'completion_tokens_est': completion_tokens,
        'total_tokens_est': prompt_tokens + completion_tokens,
        'estimated_cost_usd': round(cost_usd, 4),
        'estimation_method': 'tiktoken' if TIKTOKEN_AVAILABLE else 'character_count_fallback',
        'note': 'Using o200k_base encoding (gpt-4o/gpt-5.1). Completion ~15k tokens. Cost based on gpt-4o pricing.'
    }


def create_timestamped_output_dir(pdf_path: str, base_output_dir: str) -> str:
    """
    Create timestamped output directory based on PDF filename
    
    Args:
        pdf_path: Path to input PDF
        base_output_dir: Base output directory (e.g., 'data/test_results')
    
    Returns:
        Path to timestamped output directory (e.g., 'data/test_results/SS_01_20250114_123456')
    """
    pdf_stem = Path(pdf_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"{pdf_stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    return str(output_dir)


def step1_extract_abox(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 1: Extract ABox from PDF using OneShotTripleExtractor
    
    Input dict keys:
        - pdf_path: str
        - ontology_path: str
        - api_key: str
        - output_dir: str
        - model_name: str (optional)
        - max_pages: int (optional)
        - dpi: int (optional)
    
    Output dict keys:
        - pdf_path: str (passthrough)
        - ontology_path: str (passthrough)
        - output_dir: str (passthrough)
        - abox_path: str (generated)
        - extraction_success: bool
        - step1_duration_seconds: float (added)
        - step1_token_info: dict (added)
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1: EXTRACTING ABOX FROM PDF")
    logger.info("="*70)
    
    step1_start_time = time.time()
    
    pdf_path = input_dict['pdf_path']
    ontology_path = input_dict['ontology_path']
    api_key = input_dict['api_key']
    output_dir = input_dict['output_dir']
    model_name = input_dict.get('model_name', 'gpt-5.1')
    max_pages = input_dict.get('max_pages', None)
    dpi = input_dict.get('dpi', 300)
    
    # Initialize extractor
    try:
        extractor = OneShotTripleExtractor(
            ontology_path=ontology_path,
            api_key=api_key,
            model_name=model_name
        )
        logger.info(f"Initialized OneShotTripleExtractor with model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        step1_duration = time.time() - step1_start_time
        return {
            **input_dict,
            'abox_path': None,
            'extraction_success': False,
            'error': str(e),
            'step1_duration_seconds': step1_duration,
            'step1_token_info': {'model': model_name, 'error': str(e)}
        }
    
    # Extract triples
    try:
        # Estimate tokens before extraction
        token_info = {'model': model_name}
        
        try:
            import fitz  # PyMuPDF
            pdf_doc = fitz.open(pdf_path)
            num_pages = len(pdf_doc) if max_pages is None else min(len(pdf_doc), max_pages)
            pdf_doc.close()
            
            # Estimate tokens (assume ~25k chars for prompt text)
            token_estimates = estimate_pdf_extraction_tokens(
                num_pages=num_pages,
                dpi=dpi,
                prompt_length_chars=25000  # Typical prompt with ontology schema
            )
            
            token_info.update(token_estimates)
            
            logger.info(f"\n📊 Token Estimation:")
            logger.info(f"   Model: {model_name} (using o200k_base encoding)")
            logger.info(f"   Pages: {num_pages} @ {dpi} DPI ({token_estimates['image_resolution']})")
            logger.info(f"   Prompt tokens (est):")
            logger.info(f"     - Text: ~{token_estimates['prompt_text_tokens_est']:,}")
            logger.info(f"     - Images: ~{token_estimates['prompt_image_tokens_est']:,} ({token_estimates['tokens_per_image']} per page)")
            logger.info(f"     - Total: ~{token_estimates['prompt_tokens_est']:,}")
            logger.info(f"   Completion tokens (est): ~{token_estimates['completion_tokens_est']:,}")
            logger.info(f"   Total tokens (est): ~{token_estimates['total_tokens_est']:,}")
            logger.info(f"   💰 Estimated cost: ${token_estimates['estimated_cost_usd']:.4f} USD (gpt-4o pricing)")
            logger.info(f"   Note: Cost estimate uses gpt-4o pricing. Adjust for actual model pricing if different.")
            
        except Exception as e:
            logger.warning(f"⚠️  Token estimation failed: {e}")
            token_info['note'] = f'Estimation failed: {str(e)}'
        
        success = extractor.extract_from_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            max_pages=max_pages,
            dpi=dpi
        )
        
        step1_duration = time.time() - step1_start_time
        
        # Try to extract token usage if available from response
        # Note: LangChain's ChatOpenAI doesn't automatically track tokens
        # We'll need to manually calculate or extract from response metadata
        token_info = {
            'model': model_name,
            'note': 'Token usage tracking requires response metadata access'
        }
        
        if not success:
            logger.error("Extraction failed")
            return {
                **input_dict,
                'abox_path': None,
                'extraction_success': False,
                'error': 'Extraction returned False',
                'step1_duration_seconds': step1_duration,
                'step1_token_info': token_info
            }
        
        # Construct expected ABox path
        pdf_basename = Path(pdf_path).stem
        abox_path = Path(output_dir) / f"{pdf_basename}_extracted_abox.ttl"
        
        if not abox_path.exists():
            logger.error(f"Expected ABox file not found: {abox_path}")
            return {
                **input_dict,
                'abox_path': None,
                'extraction_success': False,
                'error': f'ABox file not found: {abox_path}',
                'step1_duration_seconds': step1_duration,
                'step1_token_info': token_info
            }
        
        logger.info(f"Successfully extracted ABox: {abox_path}")
        logger.info(f"Step 1 duration: {step1_duration:.2f} seconds")
        
        return {
            **input_dict,
            'abox_path': str(abox_path),
            'extraction_success': True,
            'step1_duration_seconds': step1_duration,
            'step1_token_info': token_info
        }
        
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        step1_duration = time.time() - step1_start_time
        return {
            **input_dict,
            'abox_path': None,
            'extraction_success': False,
            'error': str(e),
            'step1_duration_seconds': step1_duration,
            'step1_token_info': {}
        }


def step2_merge_ontology(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Merge TBox and ABox using SimpleTTLMerger
    
    Input dict keys:
        - ontology_path: str (TBox)
        - abox_path: str (from step 1)
        - output_dir: str
        - extraction_success: bool (from step 1)
        - validate_classes: bool (optional)
    
    Output dict keys:
        - merged_ttl_path: str (generated)
        - merge_success: bool
        - pipeline_success: bool (overall)
        - step2_duration_seconds: float (added)
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 2: MERGING TBOX AND ABOX")
    logger.info("="*70)
    
    step2_start_time = time.time()
    
    # Check if extraction succeeded
    if not input_dict.get('extraction_success', False):
        logger.error("Cannot proceed with merge - extraction failed")
        step2_duration = time.time() - step2_start_time
        return {
            **input_dict,
            'merged_ttl_path': None,
            'merge_success': False,
            'pipeline_success': False,
            'error': 'Extraction step failed',
            'step2_duration_seconds': step2_duration
        }
    
    tbox_path = input_dict['ontology_path']
    abox_path = input_dict['abox_path']
    output_dir = input_dict['output_dir']
    validate_classes = input_dict.get('validate_classes', True)
    
    # Construct output path for merged ontology
    pdf_basename = Path(input_dict['pdf_path']).stem
    merged_ttl_path = Path(output_dir) / f"{pdf_basename}_merged_ontology.ttl"
    
    try:
        success = merge_ttl_files(
            tbox_path=tbox_path,
            abox_path=abox_path,
            output_path=str(merged_ttl_path),
            validate_classes=validate_classes
        )
        
        step2_duration = time.time() - step2_start_time
        
        if not success:
            logger.error("Merge failed")
            return {
                **input_dict,
                'merged_ttl_path': None,
                'merge_success': False,
                'pipeline_success': False,
                'error': 'Merge returned False',
                'step2_duration_seconds': step2_duration
            }
        
        if not merged_ttl_path.exists():
            logger.error(f"Expected merged file not found: {merged_ttl_path}")
            return {
                **input_dict,
                'merged_ttl_path': None,
                'merge_success': False,
                'pipeline_success': False,
                'error': f'Merged file not found: {merged_ttl_path}',
                'step2_duration_seconds': step2_duration
            }
        
        logger.info(f"Successfully created merged ontology: {merged_ttl_path}")
        logger.info(f"Step 2 duration: {step2_duration:.2f} seconds")
        
        return {
            **input_dict,
            'merged_ttl_path': str(merged_ttl_path),
            'merge_success': True,
            'pipeline_success': True,
            'step2_duration_seconds': step2_duration
        }
        
    except Exception as e:
        logger.error(f"Merge error: {e}", exc_info=True)
        step2_duration = time.time() - step2_start_time
        return {
            **input_dict,
            'merged_ttl_path': None,
            'merge_success': False,
            'pipeline_success': False,
            'error': str(e),
            'step2_duration_seconds': step2_duration
        }


def save_performance_metrics(result: Dict[str, Any], output_dir: str) -> str:
    """
    Save performance metrics and evaluation data to JSON file
    
    Args:
        result: Result dictionary from pipeline execution
        output_dir: Output directory path
    
    Returns:
        Path to saved metrics file
    """
    pdf_basename = Path(result['pdf_path']).stem
    metrics_path = Path(output_dir) / f"{pdf_basename}_performance_metrics.json"
    
    # Calculate total duration
    step1_duration = result.get('step1_duration_seconds', 0)
    step2_duration = result.get('step2_duration_seconds', 0)
    total_duration = step1_duration + step2_duration
    
    # Get file sizes
    file_sizes = {}
    if result.get('abox_path') and Path(result['abox_path']).exists():
        file_sizes['abox_ttl_kb'] = Path(result['abox_path']).stat().st_size / 1024
    
    if result.get('merged_ttl_path') and Path(result['merged_ttl_path']).exists():
        file_sizes['merged_ttl_kb'] = Path(result['merged_ttl_path']).stat().st_size / 1024
    
    # Get PDF info
    pdf_size_kb = Path(result['pdf_path']).stat().st_size / 1024
    
    # Get token info
    token_info = result.get('step1_token_info', {})
    
    # Compile metrics
    metrics = {
        'pipeline_info': {
            'pdf_filename': Path(result['pdf_path']).name,
            'pdf_size_kb': round(pdf_size_kb, 2),
            'ontology_path': result.get('ontology_path', ''),
            'model_name': result.get('model_name', 'gpt-5'),
            'dpi': result.get('dpi', 300),
            'max_pages': result.get('max_pages', 'all'),
            'execution_timestamp': datetime.now().isoformat()
        },
        'execution_time': {
            'step1_extraction_seconds': round(step1_duration, 2),
            'step2_merge_seconds': round(step2_duration, 2),
            'total_pipeline_seconds': round(total_duration, 2),
            'total_pipeline_minutes': round(total_duration / 60, 2)
        },
        'token_usage_estimation': token_info,
        'output_files': {
            'abox_path': result.get('abox_path', ''),
            'merged_ttl_path': result.get('merged_ttl_path', ''),
            'file_sizes': file_sizes
        },
        'status': {
            'extraction_success': result.get('extraction_success', False),
            'merge_success': result.get('merge_success', False),
            'pipeline_success': result.get('pipeline_success', False),
            'error': result.get('error', None)
        }
    }
    
    # Save to JSON
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*70}")
    logger.info("PERFORMANCE METRICS SAVED")
    logger.info(f"{'='*70}")
    logger.info(f"Metrics file: {metrics_path}")
    logger.info(f"\nExecution Summary:")
    logger.info(f"  Total time: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    logger.info(f"  - Step 1 (Extraction): {step1_duration:.2f}s")
    logger.info(f"  - Step 2 (Merge): {step2_duration:.2f}s")
    
    if token_info.get('total_tokens_est'):
        logger.info(f"\nToken Usage (Estimated):")
        logger.info(f"  Total: ~{token_info['total_tokens_est']:,} tokens")
        logger.info(f"  - Prompt: ~{token_info.get('prompt_tokens_est', 0):,}")
        logger.info(f"  - Completion: ~{token_info.get('completion_tokens_est', 0):,}")
        if token_info.get('estimated_cost_usd', 0) > 0:
            logger.info(f"  Cost: ~${token_info['estimated_cost_usd']:.4f} USD")
    
    if file_sizes:
        logger.info(f"\nOutput Files:")
        for key, size in file_sizes.items():
            logger.info(f"  - {key}: {size:.2f} KB")
    
    logger.info(f"{'='*70}\n")
    
    return str(metrics_path)


def create_pdf_to_ttl_chain():
    """
    Create the complete PDF to TTL chain using LangChain Expression Language
    
    Returns:
        Runnable chain that processes PDF to final merged TTL
    """
    # Step 1: Extract ABox from PDF
    extract_abox_step = RunnableLambda(step1_extract_abox)
    
    # Step 2: Merge TBox and ABox
    merge_ontology_step = RunnableLambda(step2_merge_ontology)
    
    # Chain: extract -> merge
    chain = extract_abox_step | merge_ontology_step
    
    return chain


def main():
    """Main execution function"""
    import argparse

    load_dotenv()

    try:
        project_root = Path(__file__).parent.parent.parent
    except NameError:
        project_root = Path.cwd()

    parser = argparse.ArgumentParser(
        description="One-shot PDF to TTL extraction chain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        default=str(project_root / "data" / "test_materials" / "SS_03.pdf"),
        help="Path to input PDF file",
    )
    parser.add_argument(
        "--ontology",
        default=str(project_root / "data" / "ontology" / "DeviceDimension_demo.rdf"),
        help="Path to TBox .rdf file",
    )
    parser.add_argument(
        "--model", default="gpt-5.1",
        help="OpenAI model name",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="PDF page rendering resolution (higher = more detail, more tokens)",
    )
    parser.add_argument(
        "--max_pages", type=int, default=None,
        help="Maximum number of pages to process (omit to process all pages)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Explicit output directory. Auto-creates a timestamped directory under "
             "data/test_results/one_shot_extraction/ when omitted.",
    )
    parser.add_argument(
        "--no_validate", action="store_true",
        help="Skip class validation against the TBox after merge",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set — add it to .env or export it as an env var")
        sys.exit(1)

    pdf_path = Path(args.pdf)
    ontology_path = Path(args.ontology)
    base_output_dir = project_root / "data" / "test_results" / "one_shot_extraction"

    # Validate input files
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    if not ontology_path.exists():
        logger.error(f"Ontology file not found: {ontology_path}")
        sys.exit(1)

    # Create timestamped output directory
    output_dir = args.output_dir if args.output_dir else create_timestamped_output_dir(
        pdf_path=str(pdf_path),
        base_output_dir=str(base_output_dir),
    )

    # Prepare input dictionary
    input_dict = {
        'pdf_path': str(pdf_path),
        'ontology_path': str(ontology_path),
        'api_key': api_key,
        'output_dir': output_dir,
        'model_name': args.model,
        'max_pages': args.max_pages,
        'dpi': args.dpi,
        'validate_classes': not args.no_validate,
    }
    
    # Create and run chain
    logger.info("\n" + "="*70)
    logger.info("STARTING PDF TO TTL PIPELINE")
    logger.info("="*70)
    logger.info(f"Input PDF: {pdf_path}")
    logger.info(f"Ontology: {ontology_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70)
    
    # Track total pipeline time
    pipeline_start_time = time.time()
    
    chain = create_pdf_to_ttl_chain()
    
    try:
        result = chain.invoke(input_dict)
        
        pipeline_total_time = time.time() - pipeline_start_time
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE EXECUTION COMPLETED")
        logger.info("="*70)
        logger.info(f"Extraction success: {result.get('extraction_success', False)}")
        logger.info(f"Merge success: {result.get('merge_success', False)}")
        logger.info(f"Overall success: {result.get('pipeline_success', False)}")
        logger.info(f"Total wall-clock time: {pipeline_total_time:.2f}s ({pipeline_total_time/60:.2f} min)")
        
        # Save performance metrics
        if result.get('pipeline_success', False):
            logger.info(f"\nFinal output: {result['merged_ttl_path']}")
            
            # Save metrics
            metrics_path = save_performance_metrics(result, output_dir)
            
            logger.info("\n✓ Pipeline completed successfully!")
            logger.info(f"✓ Performance metrics saved: {metrics_path}")
        else:
            logger.error(f"\n✗ Pipeline failed: {result.get('error', 'Unknown error')}")
            
            # Save metrics even for failed runs
            metrics_path = save_performance_metrics(result, output_dir)
            logger.info(f"Performance metrics (partial): {metrics_path}")
            
            sys.exit(1)
        
        logger.info("="*70)
        
    except Exception as e:
        pipeline_total_time = time.time() - pipeline_start_time
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        logger.error(f"Failed after {pipeline_total_time:.2f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()