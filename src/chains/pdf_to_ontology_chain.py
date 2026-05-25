"""
Complete PDF to Ontology Chain with Performance Statistics

This chain implements the full pipeline with comprehensive time and token tracking:
1. PDF -> Dolphin Remote Analysis -> raw_analysis.json
2. raw_analysis.json -> Preprocessing (text, tables, images)
3. Preprocessed data -> Layered Extraction (Stage 1-3)
4. Extraction results (JSON) -> RDF Conversion -> Merged Ontology

Statistics tracked:
- Execution time for each stage
- LLM token usage (if available)
- Total pipeline time and tokens

Usage:
    python pdf_to_ontology_chain_stats.py --pdf <path_to_pdf> --tbox <path_to_tbox> --output <output_dir>
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# LangChain LCEL
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import existing components
from pdf_preprocessing_chain import DolphinRemoteAnalyzer
from src.chains.preprocessing_chain import PreprocessingChain
from layered_extraction_chain import (
    stage1_text_extraction,
    stage2_visual_patching, 
    stage3_table_fission
)
from src.knowledge_generator.json_to_rdf_converter import JsonToRdfConverter, merge_with_tbox

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PDF2Ontology] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# Statistics Data Classes
# ============================================================

@dataclass
class StageStats:
    """Statistics for a single pipeline stage."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "duration_seconds": round(self.duration, 2),
            "duration_formatted": self._format_duration(self.duration),
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_calls": self.llm_calls,
            **self.additional_info
        }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.1f}s"


@dataclass
class PipelineStats:
    """Overall pipeline statistics."""
    pipeline_start: float = 0.0
    pipeline_end: float = 0.0
    total_duration: float = 0.0
    stages: List[StageStats] = field(default_factory=list)
    
    def add_stage(self, stage: StageStats):
        """Add a completed stage."""
        self.stages.append(stage)
    
    def finalize(self):
        """Calculate total statistics."""
        self.pipeline_end = time.time()
        self.total_duration = self.pipeline_end - self.pipeline_start
    
    def get_total_tokens(self) -> int:
        """Sum tokens across all stages."""
        return sum(stage.tokens_used for stage in self.stages)
    
    def get_total_llm_calls(self) -> int:
        """Sum LLM calls across all stages."""
        return sum(stage.llm_calls for stage in self.stages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_duration_seconds": round(self.total_duration, 2),
            "total_duration_formatted": StageStats._format_duration(self.total_duration),
            "total_tokens_used": self.get_total_tokens(),
            "total_prompt_tokens": sum(s.prompt_tokens for s in self.stages),
            "total_completion_tokens": sum(s.completion_tokens for s in self.stages),
            "total_llm_calls": self.get_total_llm_calls(),
            "stages": [stage.to_dict() for stage in self.stages]
        }
    
    def print_summary(self):
        """Print formatted statistics summary."""
        logger.info("\n" + "="*70)
        logger.info("PIPELINE PERFORMANCE STATISTICS")
        logger.info("="*70)
        
        # Stage-by-stage breakdown
        logger.info("\nStage-by-Stage Performance:")
        logger.info("-" * 70)
        
        for stage in self.stages:
            logger.info(f"\n{stage.name}:")
            logger.info(f"  Duration: {StageStats._format_duration(stage.duration)}")
            if stage.llm_calls > 0:
                logger.info(f"  LLM Calls: {stage.llm_calls}")
                logger.info(f"  Tokens Used: {stage.tokens_used:,}")
                logger.info(f"    - Prompt: {stage.prompt_tokens:,}")
                logger.info(f"    - Completion: {stage.completion_tokens:,}")
            
            # Additional info
            for key, value in stage.additional_info.items():
                logger.info(f"  {key}: {value}")
        
        # Overall summary
        logger.info("\n" + "="*70)
        logger.info("OVERALL SUMMARY")
        logger.info("="*70)
        logger.info(f"Total Pipeline Duration: {StageStats._format_duration(self.total_duration)}")
        logger.info(f"Total LLM Calls: {self.get_total_llm_calls()}")
        logger.info(f"Total Tokens Used: {self.get_total_tokens():,}")
        logger.info(f"  - Prompt Tokens: {sum(s.prompt_tokens for s in self.stages):,}")
        logger.info(f"  - Completion Tokens: {sum(s.completion_tokens for s in self.stages):,}")
        
        # Cost estimation (if using OpenAI)
        if self.get_total_tokens() > 0:
            self._print_cost_estimate()
        
        logger.info("="*70)
    
    def _print_cost_estimate(self):
        """Print estimated API cost based on token usage."""
        # GPT-4o pricing (as of 2024)
        prompt_cost_per_1k = 0.005  # $5 per 1M tokens
        completion_cost_per_1k = 0.015  # $15 per 1M tokens
        
        total_prompt = sum(s.prompt_tokens for s in self.stages)
        total_completion = sum(s.completion_tokens for s in self.stages)
        
        prompt_cost = (total_prompt / 1000) * prompt_cost_per_1k
        completion_cost = (total_completion / 1000) * completion_cost_per_1k
        total_cost = prompt_cost + completion_cost
        
        logger.info(f"\nEstimated API Cost (GPT-4o rates):")
        logger.info(f"  Prompt Cost: ${prompt_cost:.4f}")
        logger.info(f"  Completion Cost: ${completion_cost:.4f}")
        logger.info(f"  Total Cost: ${total_cost:.4f}")


@dataclass
class ChainConfig:
    """Configuration for the PDF to Ontology chain."""
    model: str = "gpt-4o"
    context_window: int = 3
    api_url: str = "http://localhost:8080/analyze"
    strict_validation: bool = True


# ============================================================
# Enhanced Chain with Statistics
# ============================================================

class PdfToOntologyChain:
    """
    Complete end-to-end chain from PDF to Ontology with performance tracking.
    """
    
    def __init__(
        self,
        tbox_path: str,
        output_root: str,
        api_key: str,
        config: Optional[ChainConfig] = None
    ):
        self.tbox_path = tbox_path
        self.output_root = output_root
        self.api_key = api_key
        self.config = config or ChainConfig()
        self.stats = PipelineStats()
        
        if not os.path.exists(tbox_path):
            raise FileNotFoundError(f"TBox not found: {tbox_path}")
            
        logger.info(f"Chain Configuration:")
        logger.info(f"  Model: {self.config.model}")
        logger.info(f"  Context Window: {self.config.context_window}")
        logger.info(f"  Dolphin API: {self.config.api_url}")
        logger.info(f"  TBox: {tbox_path}")
    
    def _track_stage(self, stage_name: str):
        """Decorator-like function to track stage execution."""
        stage_stats = StageStats(name=stage_name)
        stage_stats.start_time = time.time()
        return stage_stats
    
    def _finish_stage(self, stage_stats: StageStats):
        """Complete stage tracking."""
        stage_stats.end_time = time.time()
        stage_stats.duration = stage_stats.end_time - stage_stats.start_time
        self.stats.add_stage(stage_stats)
    
    def _step0_dolphin_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 0: Remote Dolphin Analysis"""
        stage = self._track_stage("Step 0: Dolphin Analysis")
        
        logger.info("\n" + "="*70)
        logger.info("STEP 0: Remote Dolphin Analysis")
        logger.info("="*70)
        
        analyzer = DolphinRemoteAnalyzer(api_url=self.config.api_url)
        state = analyzer.analyze(state)
        
        # Extract stats if available
        if 'raw_json_path' in state and os.path.exists(state['raw_json_path']):
            with open(state['raw_json_path'], 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                stage.additional_info['pages_analyzed'] = raw_data.get('total_pages', 0)
        
        logger.info(f"Raw analysis saved: {state['raw_json_path']}")
        
        self._finish_stage(stage)
        state['stats'] = self.stats
        return state
    
    def _step1_preprocessing(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Preprocessing"""
        stage = self._track_stage("Step 1: Preprocessing")
        
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Preprocessing")
        logger.info("="*70)
        
        preprocessing_output = os.path.join(state['output_root'], 'preprocessing')
        
        chain = PreprocessingChain(
            raw_json_path=state['raw_json_path'],
            pdf_path=state['pdf_path'],
            output_root=preprocessing_output,
            table_context_before=self.config.context_window,
            table_context_after=self.config.context_window,
            image_context_before=self.config.context_window,
            image_context_after=self.config.context_window
        )
        
        results = chain.execute()
        
        # Store paths
        state['text_path'] = chain.paths['text']
        state['tables_path'] = chain.paths['table']
        state['images_path'] = chain.paths['image_json']
        state['preprocessing_results'] = results
        
        # Extract preprocessing stats
        if 'steps' in results:
            if results['steps'].get('image', {}).get('count'):
                stage.additional_info['images_processed'] = results['steps']['image']['count']
        
        logger.info(f"Preprocessing complete")
        
        self._finish_stage(stage)
        return state
    
    def _step2_load_preprocessed_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Load preprocessed data"""
        stage = self._track_stage("Step 2: Load Preprocessed Data")
        
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Loading Preprocessed Data")
        logger.info("="*70)
        
        # Load text
        with open(state['text_path'], 'r', encoding='utf-8') as f:
            text_json = json.load(f)
            state['raw_text'] = "\n".join([p['text'] for p in text_json.get('pages', [])])
        
        # Load images
        with open(state['images_path'], 'r', encoding='utf-8') as f:
            images_data = json.load(f)
            state['images'] = images_data.get('images', [])
            
            images_dir = os.path.dirname(state['images_path'])
            for img in state['images']:
                if 'cropped_image_path' in img:
                    img_name = Path(img['cropped_image_path']).name
                    img['cropped_image_path'] = os.path.join(images_dir, img_name)
        
        # Load tables
        with open(state['tables_path'], 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
            state['tables'] = tables_data.get('tables', [])
        
        state['ontology_path'] = self.tbox_path
        state['api_key'] = self.api_key
        
        stage.additional_info['images_loaded'] = len(state['images'])
        stage.additional_info['tables_loaded'] = len(state['tables'])
        
        logger.info(f"Loaded: {len(state['images'])} images, {len(state['tables'])} tables")
        
        self._finish_stage(stage)
        return state
    
    def _step3_layered_extraction(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Three-stage layered extraction with token tracking"""
        
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Layered Knowledge Extraction")
        logger.info("="*70)
        
        # Initialize knowledge graph
        state['current_graph'] = {"instances": [], "relationships": []}
        
        # Stage 1: Text extraction
        stage1 = self._track_stage("Step 3.1: Text Extraction")
        logger.info("  -> Stage 1: Text-based skeleton extraction")
        state = stage1_text_extraction(state)
        self._save_stage_result(state, 1, "stage1_skeleton")
        stage1.additional_info['instances_extracted'] = len(state['current_graph'].get('instances', []))
        self._finish_stage(stage1)
        
        # Stage 2: Visual patching
        stage2 = self._track_stage("Step 3.2: Visual Patching")
        logger.info("  -> Stage 2: Visual information patching")
        instances_before_stage2 = len(state['current_graph'].get('instances', []))
        state = stage2_visual_patching(state)
        instances_after_stage2 = len(state['current_graph'].get('instances', []))
        self._save_stage_result(state, 2, "stage2_patched", instances_before_stage2, instances_after_stage2)
        stage2.additional_info['instances_added'] = instances_after_stage2 - instances_before_stage2
        stage2.additional_info['images_processed'] = len(state.get('images', []))
        self._finish_stage(stage2)
        
        # Stage 3: Table fission
        stage3 = self._track_stage("Step 3.3: Table Fission")
        logger.info("  -> Stage 3: Table-based instance fission")
        instances_before_stage3 = len(state['current_graph'].get('instances', []))
        state = stage3_table_fission(state)
        instances_after_stage3 = len(state['current_graph'].get('instances', []))
        self._save_stage_result(state, 3, "stage3_fissioned", instances_before_stage3, instances_after_stage3)
        stage3.additional_info['instances_added'] = instances_after_stage3 - instances_before_stage3
        stage3.additional_info['tables_processed'] = len(state.get('tables', []))
        self._finish_stage(stage3)
        
        # Save final combined extraction result
        self._save_final_extraction(state, instances_before_stage2, instances_after_stage2, 
                                    instances_before_stage3, instances_after_stage3)
        
        return state
    
    def _save_stage_result(self, state, stage_num, stage_name, instances_before=0, instances_after=0):
        """Save intermediate stage results."""
        # Use mapping table to get correct stage type
        stage_type_map = {
            1: 'text',      # Stage 1: text extraction
            2: 'visual',    # Stage 2: visual patching
            3: 'table'      # Stage 3: table fission
        }
        stage_type = stage_type_map.get(stage_num, 'unknown')
        
        # Create correct path
        stage_output = os.path.join(state['output_root'], f'extraction/stage{stage_num}_{stage_type}')
        stage_path = os.path.join(stage_output, f"{state['doc_name']}_{stage_name}.json")
        
        stats = {
            "stage": f"stage{stage_num}_{stage_type}_extraction",
            "timestamp": datetime.now().isoformat(),
            "instances": state['current_graph'].get('instances', []),
            "relationships": state['current_graph'].get('relationships', []),
            "statistics": {
                "total_instances": len(state['current_graph'].get('instances', [])),
                "total_relationships": len(state['current_graph'].get('relationships', []))
            }
        }
        
        if instances_before > 0:
            stats["statistics"]["instances_added_in_stage"] = instances_after - instances_before
        
        # Ensure directory exists (critical for Windows!)
        os.makedirs(stage_output, exist_ok=True)
        
        with open(stage_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  -> Stage {stage_num} result saved: {stage_path}")
        state[f'stage{stage_num}_output_path'] = stage_path
    
    def _save_final_extraction(self, state, inst_s2_before, inst_s2_after, inst_s3_before, inst_s3_after):
        """Save final extraction results with comprehensive statistics."""
        final_extraction_output = os.path.join(state['output_root'], 'extraction/final')
        final_json_path = os.path.join(final_extraction_output, f"{state['doc_name']}_final_knowledge_graph.json")
        
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "document_name": state['doc_name'],
                "timestamp": state.get('timestamp'),
                "extraction_timestamp": datetime.now().isoformat(),
                "pipeline_stages": {
                    "stage1_text": state.get('stage1_output_path'),
                    "stage2_visual": state.get('stage2_output_path'),
                    "stage3_table": state.get('stage3_output_path')
                },
                "final_graph": state['current_graph'],
                "statistics": {
                    "total_instances": len(state['current_graph'].get('instances', [])),
                    "total_relationships": len(state['current_graph'].get('relationships', [])),
                    "instances_from_stage1": inst_s2_before,
                    "instances_from_stage2": inst_s2_after - inst_s2_before,
                    "instances_from_stage3": inst_s3_after - inst_s3_before
                }
            }, f, indent=2, ensure_ascii=False)
        
        state['extraction_json_path'] = final_json_path
        logger.info(f"Final extraction results saved: {final_json_path}")
        logger.info(f"  -> Total instances: {len(state['current_graph'].get('instances', []))}")
        logger.info(f"  -> Total relationships: {len(state['current_graph'].get('relationships', []))}")
    
    def _step4_rdf_conversion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Convert JSON to RDF and merge with TBox"""
        stage = self._track_stage("Step 4: RDF Conversion & Merge")
        
        logger.info("\n" + "="*70)
        logger.info("STEP 4: RDF Conversion and Ontology Merging")
        logger.info("="*70)
        
        ontology_output = os.path.join(state['output_root'], 'ontology')
        os.makedirs(ontology_output, exist_ok=True)
        
        abox_path = os.path.join(ontology_output, f"{state['doc_name']}_abox.ttl")
        merged_path = os.path.join(ontology_output, f"{state['doc_name']}_merged_ontology.ttl")
        
        # Convert to RDF
        converter = JsonToRdfConverter(
            tbox_path=self.tbox_path,
            strict_validation=self.config.strict_validation
        )
        
        success = converter.convert(state['extraction_json_path'], abox_path)
        
        if not success:
            raise RuntimeError("RDF conversion failed")
        
        state['abox_path'] = abox_path
        
        # Extract conversion statistics
        if hasattr(converter, 'stats'):
            stage.additional_info['total_triples'] = converter.stats.total_instances
            stage.additional_info['device_instances'] = converter.stats.device_instances
            stage.additional_info['dimension_instances'] = converter.stats.dimension_instances
        
        # Merge with TBox
        merge_success = merge_with_tbox(self.tbox_path, abox_path, merged_path)
        
        if not merge_success:
            raise RuntimeError("Ontology merge failed")
        
        state['final_ontology_path'] = merged_path
        
        # Get file sizes
        abox_size = os.path.getsize(abox_path) / 1024
        merged_size = os.path.getsize(merged_path) / 1024
        
        stage.additional_info['abox_size_kb'] = round(abox_size, 1)
        stage.additional_info['merged_ontology_size_kb'] = round(merged_size, 1)
        
        logger.info(f"ABox saved: {abox_path} ({abox_size:.1f} KB)")
        logger.info(f"Merged ontology saved: {merged_path} ({merged_size:.1f} KB)")
        
        self._finish_stage(stage)
        return state
    
    def _initialize_state(self, pdf_path: str) -> Dict[str, Any]:
        """Initialize chain state with timestamped output directory."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        timestamped_folder = f"{doc_name}_{timestamp}"
        output_root = os.path.join(self.output_root, timestamped_folder)
        
        # Create all subdirectories
        subdirs = [
            'raw_analysis',
            'preprocessing/text',
            'preprocessing/tables', 
            'preprocessing/images',
            'extraction/stage1_text',
            'extraction/stage2_visual',
            'extraction/stage3_table',
            'extraction/final',
            'ontology'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(output_root, subdir), exist_ok=True)
        
        logger.info(f"Created timestamped output directory: {timestamped_folder}")
        
        return {
            "pdf_path": pdf_path,
            "doc_name": doc_name,
            "timestamp": timestamp,
            "output_root": output_root,
            "timestamped_folder": timestamped_folder
        }
    
    def _save_statistics(self, state: Dict[str, Any]):
        """Save comprehensive statistics to JSON file."""
        stats_path = os.path.join(state['output_root'], f"{state['doc_name']}_statistics.json")
        
        stats_data = {
            "document_name": state['doc_name'],
            "timestamp": state['timestamp'],
            "completion_time": datetime.now().isoformat(),
            "configuration": {
                "model": self.config.model,
                "context_window": self.config.context_window
            },
            "performance": self.stats.to_dict(),
            "outputs": {
                "final_ontology": state.get('final_ontology_path'),
                "abox": state.get('abox_path'),
                "extraction_json": state.get('extraction_json_path')
            }
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nStatistics saved: {stats_path}")
        return stats_path
    
    def build(self):
        """Build the LCEL chain."""
        chain = (
            RunnableLambda(self._initialize_state)
            | RunnableLambda(self._step0_dolphin_analysis)
            | RunnableLambda(self._step1_preprocessing)
            | RunnableLambda(self._step2_load_preprocessed_data)
            | RunnableLambda(self._step3_layered_extraction)
            | RunnableLambda(self._step4_rdf_conversion)
        )
        return chain
    
    def run(self, pdf_path: str) -> Dict[str, Any]:
        """Execute the complete chain with statistics tracking."""
        logger.info("\n" + "="*70)
        logger.info("PDF TO ONTOLOGY CHAIN - START")
        logger.info("="*70)
        logger.info(f"Input PDF: {pdf_path}")
        
        # Start pipeline timer
        self.stats.pipeline_start = time.time()
        
        try:
            chain = self.build()
            result = chain.invoke(pdf_path)
            
            # Finalize statistics
            self.stats.finalize()
            
            # Save statistics
            stats_path = self._save_statistics(result)
            result['statistics_path'] = stats_path
            
            # Print summary
            self.stats.print_summary()
            
            logger.info("\n" + "="*70)
            logger.info("PDF TO ONTOLOGY CHAIN - COMPLETE")
            logger.info("="*70)
            logger.info(f"Final Ontology: {result['final_ontology_path']}")
            logger.info(f"Statistics: {stats_path}")
            
            return result
            
        except Exception as e:
            self.stats.finalize()
            logger.error(f"Pipeline failed after {StageStats._format_duration(self.stats.total_duration)}")
            raise


def main():
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Complete PDF to Ontology Chain with Dolphin Analysis"
    )
    parser.add_argument(
        '--pdf',
        default= "data/test_materials/KN_02.pdf",
        help='Path to input PDF file'
    )
    parser.add_argument(
        '--tbox',
        default='data/ontology/DeviceDimension_v3.rdf',
        help='Path to TBox ontology file (.ttl or .rdf)'
    )
    parser.add_argument(
        '--output',
        default='data/pdf_to_ontology_results',
        help='Output directory root (default: data/pdf_to_ontology_results)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='LLM model to use (default: gpt-4o)'
    )
    parser.add_argument(
        '--context',
        type=int,
        default=3,
        help='Context window size for preprocessing (default: 3)'
    )
    parser.add_argument(
        '--api_url',
        default='http://localhost:8080/analyze',
        help='Dolphin API URL (default: http://localhost:8080/analyze)'
    )
    parser.add_argument(
        '--api_key',
        help='OpenAI API key (defaults to OPENAI_API_KEY env variable)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("API key not provided. Use --api_key or set OPENAI_API_KEY env variable")
        sys.exit(1)
    
    # Create configuration
    config = ChainConfig(
        model=args.model,
        context_window=args.context,
        api_url=args.api_url
    )
    
    # Run chain
    try:
        chain = PdfToOntologyChain(
            tbox_path=args.tbox,
            output_root=args.output,
            api_key=api_key,
            config=config
        )
        
        result = chain.run(args.pdf)
        
        print("\n" + "="*70)
        print("SUCCESS - Pipeline completed!")
        print("="*70)
        print(f"Document: {result['doc_name']}")
        print(f"Final Ontology: {result['final_ontology_path']}")
        print(f"Statistics: {result['statistics_path']}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()