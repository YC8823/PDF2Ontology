import logging
import os
import json
from typing import Optional, List, Dict, Any, Tuple

import pdfplumber
from pydantic import BaseModel

# Import from the new dedicated ontology loader
from src.preprocessors.ontology_loader import load_and_serialize_ontology

# --- Configure logging ---
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class ContentBlock(BaseModel):
    page_number: int
    block_type: str
    title: Optional[str] = None
    content: Any
    bbox: Tuple[float, float, float, float]

class ProcessedDocument(BaseModel):
    document_name: str
    content_blocks: List[ContentBlock]

# --- Core Processor Class ---
class MergedCellPDFProcessor:
    """
    An advanced PDF processor that explicitly recognizes and handles merged cells in tables.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def _table_to_merged_cell_json(self, table: pdfplumber.table.Table) -> List[Dict[str, Any]]:
        """
        Convert table to JSON with rowspan/colspan information following the described algorithm.
        """
        raw_table = table.extract()
        if not raw_table:
            return []

        row_count = len(raw_table)
        col_count = len(raw_table[0]) if row_count > 0 else 0
        
        visited = [[False for _ in range(col_count)] for _ in range(row_count)]
        cells = []

        for r in range(row_count):
            for c in range(col_count):
                if visited[r][c]:
                    continue

                cell_text = raw_table[r][c]
                
                # Calculate rowspan
                rowspan = 1
                while r + rowspan < row_count and raw_table[r + rowspan][c] is None:
                    rowspan += 1
                
                # Calculate colspan
                colspan = 1
                while c + colspan < col_count and raw_table[r][c + colspan] is None:
                    # Ensure entire row is None
                    is_merge_candidate = True
                    for i in range(rowspan):
                        if raw_table[r+i][c+colspan] is not None:
                            is_merge_candidate = False
                            break
                    if is_merge_candidate:
                        colspan += 1
                    else:
                        break

                # Update visited matrix
                for i in range(rowspan):
                    for j in range(colspan):
                        visited[r + i][c + j] = True
                
                cells.append({
                    "row": r,
                    "col": c,
                    "text": str(cell_text or '').strip(),
                    "rowspan": rowspan,
                    "colspan": colspan
                })
        return cells

    def _find_title_for_object(self, page: pdfplumber.page.Page, obj_bbox: Tuple[float, float, float, float]) -> Optional[str]:
        obj_top = obj_bbox[1]
        title_search_y_start = max(0, obj_top - 30)
        title_search_bbox = (obj_bbox[0], title_search_y_start, obj_bbox[2], obj_top)
        
        if title_search_bbox[3] > title_search_bbox[1]:
            try:
                cropped_page = page.crop(title_search_bbox)
                title_text = cropped_page.extract_text(x_tolerance=2, y_tolerance=2)
                if title_text:
                    return title_text.strip().replace('\n', ' ')
            except ValueError:
                logger.warning(f"Could not crop page for title search with bbox {title_search_bbox}")
        return None

    def _merge_cross_page_tables(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        merged_blocks = []
        i = 0
        while i < len(blocks):
            current_block = blocks[i]
            if i + 1 < len(blocks):
                next_block = blocks[i+1]
                if (current_block.block_type == "table" and 
                    next_block.block_type == "table" and
                    next_block.page_number == current_block.page_number + 1):
                    current_rows = max(cell['row'] for cell in current_block.content) + 1 if current_block.content else 0
                    current_cols = max(cell['col'] for cell in current_block.content) + 1 if current_block.content else 0
                    next_cols = max(cell['col'] for cell in next_block.content) + 1 if next_block.content else 0
                    if current_cols == next_cols and current_cols > 0:
                        logger.info(f"Detected and merged cross-page table: page {current_block.page_number} with page {next_block.page_number}.")
                        adjusted_next_content = [
                            {**cell, "row": cell['row'] + current_rows} 
                            for cell in next_block.content
                        ]
                        current_block.content.extend(adjusted_next_content)
                        i += 1
            merged_blocks.append(current_block)
            i += 1
        return merged_blocks

    def process_document(self, pdf_path: str) -> ProcessedDocument:
        doc_name = os.path.basename(pdf_path)
        logger.info(f"Starting merged-cell processing for document: {doc_name}")
        all_blocks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                logger.info(f"Processing page {page_num}...")
                tables = page.find_tables()
                for table in tables:
                    title = self._find_title_for_object(page, table.bbox)
                    table_json = self._table_to_merged_cell_json(table)
                    if table_json:
                        all_blocks.append(ContentBlock(
                            page_number=page_num, block_type="table", title=title,
                            content=table_json, bbox=table.bbox
                        ))
        all_blocks.sort(key=lambda b: (b.page_number, b.bbox[1]))
        merged_blocks = self._merge_cross_page_tables(all_blocks)
        return ProcessedDocument(document_name=doc_name, content_blocks=merged_blocks)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_pdf_path = os.path.join(project_root, "data", "inputs", "t58700en.pdf")
    output_dir = os.path.join(project_root, "data", "outputs", "merged_cell_processing")
    os.makedirs(output_dir, exist_ok=True)
    
    processor = MergedCellPDFProcessor(output_dir=output_dir)
    processed_doc = processor.process_document(test_pdf_path)
    
    output_json_path = os.path.join(output_dir, "processed_document_with_spans.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        f.write(processed_doc.model_dump_json(indent=2))
        
    logger.info(f"--- Merged-Cell Processing Complete ---")
    logger.info(f"Processed document with spans saved to: {output_json_path}")