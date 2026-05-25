"""
Stage 1 + Stage 2 + Stage 3 Extraction Chain

End-to-end pipeline connecting text extraction (Stage 1), visual extraction
(Stage 2), and table-driven instance fission (Stage 3). Data is passed directly
between stages — no shared registry, no intermediate file I/O required.

Output paths follow the project convention:
    data/test_results/stage123_extraction/{doc_name}_{timestamp}/
        {doc_name}_stage1_skeleton.json
        {doc_name}_stage2_patched.json
        {doc_name}_stage3_fissioned.json
        {doc_name}_performance_metrics.json
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    project_root_path = Path(__file__).parent.parent.parent
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
except Exception:
    project_root_path = Path.cwd()

from src.knowledge_extractor.layered_stage1_text_extractor_refactored import Stage1TextExtractor
from src.knowledge_extractor.layered_stage2_visual_extractor_refactored import execute_stage2_visual_extraction
from src.knowledge_extractor.layered_stage3_table_extractor_refactored import Stage3TableExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_timestamped_output_dir(doc_name: str, base_output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"{doc_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return str(output_dir)


def save_performance_metrics(
    doc_name: str,
    ontology_path: str,
    model_name: str,
    stage_durations: Dict[str, float],
    output_dir: str,
) -> str:
    total = sum(stage_durations.values())
    metrics = {
        "pipeline_info": {
            "doc_name": doc_name,
            "ontology_path": ontology_path,
            "model_name": model_name,
            "execution_timestamp": datetime.now().isoformat(),
        },
        "execution_time": {
            **{f"stage{k}_seconds": round(v, 2) for k, v in stage_durations.items()},
            "total_pipeline_seconds": round(total, 2),
            "total_pipeline_minutes": round(total / 60, 2),
        },
    }
    metrics_path = Path(output_dir) / f"{doc_name}_performance_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Performance metrics saved: {metrics_path}")
    return str(metrics_path)


def run_stage123_chain(
    doc_name: str,
    text_json_path: str,
    images_json_path: str,
    tables_json_path: str,
    ontology_path: str,
    api_key: str,
    project_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_name: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Run Stage 1 → Stage 2 → Stage 3 with direct data passing between stages.

    Args:
        doc_name: Document identifier (used for output file names).
        text_json_path: Path to preprocessed text JSON (Stage 1 input).
        images_json_path: Path to images JSON (Stage 2 input).
        tables_json_path: Path to tables JSON (Stage 3 input).
        ontology_path: Path to TBox ontology (.rdf).
        api_key: OpenAI API key.
        project_root: Project root path. Auto-detected if omitted.
        output_dir: Explicit output directory. Auto-created with timestamp if omitted.
        model_name: LLM model name for all stages.

    Returns:
        Dict with keys "stage1", "stage2", "stage3", "output_dir",
        "performance_metrics_path".
    """
    if project_root is None:
        try:
            project_root = Path(__file__).parent.parent.parent
        except NameError:
            project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    if output_dir is None:
        base = project_root / "data" / "test_results" / "stage123_extraction"
        output_dir = Path(create_timestamped_output_dir(doc_name, str(base)))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load text input ──────────────────────────────────────────────────
    with open(text_json_path, "r", encoding="utf-8") as f:
        text_data = json.load(f)
    plain_text = "\n\n".join([p.get("text", "") for p in text_data.get("pages", [])])
    logger.info(f"Text loaded: {len(plain_text)} chars from {Path(text_json_path).name}")

    # ── 2. Stage 1 ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"STAGE 1: Text Extraction  [{doc_name}]")
    logger.info("=" * 60)

    t1_start = time.time()
    stage1_extractor = Stage1TextExtractor(
        ontology_path=ontology_path,
        api_key=api_key,
        model_name=model_name,
    )
    stage1_result = stage1_extractor.extract(plain_text, doc_name)
    t1_duration = time.time() - t1_start

    stage1_path = output_dir / f"{doc_name}_stage1_skeleton.json"
    with open(stage1_path, "w", encoding="utf-8") as f:
        json.dump(stage1_result, f, indent=2, ensure_ascii=False)

    devices_s1 = stage1_result.get("devices", [])
    logger.info(f"Stage 1 complete: {len(devices_s1)} prototypes | saved -> {stage1_path}")

    # ── 3. Stage 2 ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"STAGE 2: Visual Extraction  [{doc_name}]")
    logger.info("=" * 60)

    t2_start = time.time()
    stage2_result = execute_stage2_visual_extraction(
        stage1_data=stage1_result,
        images_json_path=images_json_path,
        ontology_path=ontology_path,
        api_key=api_key,
        output_dir=str(output_dir),
        doc_name=doc_name,
        model_name=model_name,
    )
    t2_duration = time.time() - t2_start

    devices_s2 = stage2_result.get("devices", [])
    logger.info(f"Stage 2 complete: {len(devices_s2)} entities in enriched graph")

    # ── 4. Stage 3 ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"STAGE 3: Table Instance Fission  [{doc_name}]")
    logger.info("=" * 60)

    t3_start = time.time()
    stage3_extractor = Stage3TableExtractor(
        api_key=api_key,
        ontology_path=ontology_path,
        model_name=model_name,
    )
    stage3_result = stage3_extractor.execute(
        stage2_data=stage2_result,
        tables_path=tables_json_path,
        output_dir=str(output_dir),
        doc_name=doc_name,
    )
    t3_duration = time.time() - t3_start

    n_protos    = len(stage3_result.get("prototype_graph", {}).get("device_prototypes", []))
    n_instances = len(stage3_result.get("instance_graph", {}).get("device_instances", []))
    logger.info(
        f"Stage 3 complete: {n_protos} prototypes, {n_instances} instances | "
        f"saved -> {stage3_result.get('_metadata', {}).get('output_path', '?')}"
    )

    # ── 5. Performance metrics ───────────────────────────────────────────────
    metrics_path = save_performance_metrics(
        doc_name=doc_name,
        ontology_path=ontology_path,
        model_name=model_name,
        stage_durations={"1": t1_duration, "2": t2_duration, "3": t3_duration},
        output_dir=str(output_dir),
    )

    return {
        "stage1": stage1_result,
        "stage2": stage2_result,
        "stage3": stage3_result,
        "output_dir": str(output_dir),
        "performance_metrics_path": metrics_path,
    }


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    try:
        _project_root = Path(__file__).parent.parent.parent
    except NameError:
        _project_root = Path.cwd()

    parser = argparse.ArgumentParser(
        description="Stage 1+2+3 three-stage extraction chain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--doc", default="SS_03",
        help="Document name; expects preprocessed JSONs under data/test_intermediate_results/{doc}/",
    )
    parser.add_argument(
        "--ontology",
        default=str(_project_root / "data" / "ontology" / "DeviceDimension_demo.rdf"),
        help="Path to TBox .rdf file",
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="OpenAI model name used for all three stages",
    )
    parser.add_argument(
        "--input_dir", default=None,
        help="Base directory containing text/ images/ tables/ sub-folders. "
             "Defaults to data/test_intermediate_results/{doc}/",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Explicit output directory. Auto-creates a timestamped directory under "
             "data/test_results/stage123_extraction/ when omitted.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set — add it to .env or export it as an env var")
        sys.exit(1)

    _doc_name = args.doc
    _base_dir = Path(args.input_dir) if args.input_dir else (
        _project_root / "data" / "test_intermediate_results" / _doc_name
    )

    results = run_stage123_chain(
        doc_name=_doc_name,
        text_json_path=str(_base_dir / "text" / f"{_doc_name}_text.json"),
        images_json_path=str(_base_dir / "images" / f"{_doc_name}_images.json"),
        tables_json_path=str(_base_dir / "tables" / f"{_doc_name}_tables.json"),
        ontology_path=args.ontology,
        api_key=api_key,
        project_root=str(_project_root),
        output_dir=args.output_dir,
        model_name=args.model,
    )

    s3 = results["stage3"]
    n = len(s3.get("instance_graph", {}).get("device_instances", []))
    logger.info(f"Pipeline complete — {n} device instances")
    logger.info(f"Output directory: {results['output_dir']}")
    logger.info(f"Metrics: {results['performance_metrics_path']}")
