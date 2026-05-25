"""
Generic Table Extractor (Context-Aware Version)

Features:
1. Reads raw_analysis.json with flat structure.
2. Extracts table coordinates (bbox).
3. Uses PDFPlumber for precise table content extraction (handles merged cells).
4. Automatically converts coordinate systems (High-DPI Pixel -> PDF Points).
5. **NEW**: Collects configurable context window around each table.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import pdfplumber
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Models ---
class TableCell(BaseModel):
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1

class ContextElement(BaseModel):
    """Context element surrounding the table"""
    element_id: str
    element_type: str
    text: str
    reading_order: int
    position: str = Field(description="'before' or 'after' relative to table")

class ExtractedTable(BaseModel):
    page_number: int
    cells: List[TableCell]
    bbox: List[float]
    table_id: str
    reading_order: int
    # NEW: Context information
    context_before: List[ContextElement] = Field(default_factory=list, description="Elements before table in reading order")
    context_after: List[ContextElement] = Field(default_factory=list, description="Elements after table in reading order")

class TableExtractor:
    def __init__(
        self, 
        raw_analysis_path: str, 
        pdf_path: str,
        context_window_before: int = 3,
        context_window_after: int = 3
    ):
        """
        Args:
            raw_analysis_path: Path to raw_analysis.json
            pdf_path: Path to PDF file
            context_window_before: Number of elements to collect before table
            context_window_after: Number of elements to collect after table
        """
        self.raw_analysis_path = raw_analysis_path
        self.pdf_path = pdf_path
        self.context_window_before = context_window_before
        self.context_window_after = context_window_after
        
        with open(raw_analysis_path, 'r', encoding='utf-8') as f:
            self.raw_analysis = json.load(f)
        
        logger.info(f"Table Extractor initialized with context window: before={context_window_before}, after={context_window_after}")

    def _build_element_index(self) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
        """Builds element index adapting to flat JSON structure."""
        all_elements = []
        elements_by_page = defaultdict(list)
        
        results_per_page = self.raw_analysis.get('results_per_page', [])
        
        for page_data in results_per_page:
            page_num = page_data.get('page_number', 1)
            elements = page_data.get('elements', [])
            
            for idx, elem in enumerate(elements):
                processed_elem = {
                    'element_id': elem.get('element_id', f"p{page_num}_e{idx}"),
                    'page_number': page_num,
                    'reading_order': elem.get('reading_order', idx),
                    'bbox': elem.get('bbox') or elem.get('pixel_bbox'),
                    'content': elem.get('text', ''),
                    'type': elem.get('label', elem.get('type', 'text'))
                }
                all_elements.append(processed_elem)
                elements_by_page[page_num].append(processed_elem)
        
        all_elements_sorted = sorted(all_elements, key=lambda e: (e['page_number'], e['reading_order']))
        return all_elements_sorted, elements_by_page

    def _collect_context(
        self, 
        table_elem: Dict, 
        all_elements: List[Dict],
        global_index: int
    ) -> Tuple[List[ContextElement], List[ContextElement]]:
        """
        Collect context elements around the table.
        
        Args:
            table_elem: The table element dict
            all_elements: Sorted list of all elements
            global_index: Index of table in all_elements
            
        Returns:
            (context_before, context_after) tuple of ContextElement lists
        """
        page_num = table_elem['page_number']
        context_before = []
        context_after = []
        
        # Collect elements BEFORE table
        for offset in range(1, self.context_window_before + 1):
            idx = global_index - offset
            if idx < 0:
                break
            
            elem = all_elements[idx]
            
            # Stop if different page
            if elem['page_number'] != page_num:
                break
            
            # Skip other tables to avoid confusion
            if elem['type'] in ['tab', 'table']:
                continue
            
            context_elem = ContextElement(
                element_id=elem['element_id'],
                element_type=elem['type'],
                text=elem['content'],
                reading_order=elem['reading_order'],
                position='before'
            )
            context_before.insert(0, context_elem)  # Insert at beginning to maintain order
        
        # Collect elements AFTER table
        for offset in range(1, self.context_window_after + 1):
            idx = global_index + offset
            if idx >= len(all_elements):
                break
            
            elem = all_elements[idx]
            
            # Stop if different page
            if elem['page_number'] != page_num:
                break
            
            # Skip other tables
            if elem['type'] in ['tab', 'table']:
                continue
            
            context_elem = ContextElement(
                element_id=elem['element_id'],
                element_type=elem['type'],
                text=elem['content'],
                reading_order=elem['reading_order'],
                position='after'
            )
            context_after.append(context_elem)
        
        logger.debug(f"Table {table_elem['element_id']}: Collected {len(context_before)} elements before, {len(context_after)} after")
        
        return context_before, context_after

    def _convert_bbox(self, bbox, img_w, img_h, pdf_w, pdf_h):
        """Coordinate Conversion: Pixel -> PDF Point"""
        x0, y0, x1, y1 = bbox
        sx, sy = pdf_w / img_w, pdf_h / img_h
        return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]

    def _extract_cells_with_plumber(self, page, bbox) -> List[Dict]:
        """Uses PDFPlumber to extract table content."""
        try:
            x0, y0, x1, y1 = bbox
            crop_box = (max(0, x0-2), max(0, y0-2), min(page.width, x1+2), min(page.height, y1+2))
            cropped = page.crop(crop_box)
            
            tables = cropped.find_tables()
            if not tables:
                return []
            
            table = tables[0]
            raw_data = table.extract()
            if not raw_data:
                return []

            cells = []
            rows = len(raw_data)
            cols = len(raw_data[0]) if rows > 0 else 0
            
            for r in range(rows):
                for c in range(cols):
                    val = raw_data[r][c]
                    if val:
                        cells.append({
                            "row": r, "col": c, 
                            "text": str(val).strip(), 
                            "rowspan": 1, "colspan": 1
                        })
            return cells
        except Exception as e:
            logger.error(f"Plumber extraction error: {e}")
            return []

    def extract_and_save(self, output_path: str):
        print(f"Extracting tables with context window (before={self.context_window_before}, after={self.context_window_after})...")
        
        all_elems, _ = self._build_element_index()
        
        # Build global index map
        elem_index_map = {
            (elem['page_number'], elem['element_id']): idx 
            for idx, elem in enumerate(all_elems)
        }
        
        extracted_tables = []
        table_elems = [e for e in all_elems if e['type'] in ['tab', 'table']]
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for idx, tbl_elem in enumerate(table_elems):
                page_num = tbl_elem['page_number']
                page = pdf.pages[page_num - 1]
                
                # Get image dimensions
                page_data = next((p for p in self.raw_analysis['results_per_page'] if p['page_number'] == page_num), {})
                img_size = page_data.get('image_size', {'width': 2481, 'height': 3508})
                
                pdf_bbox = self._convert_bbox(
                    tbl_elem['bbox'], 
                    img_size['width'], img_size['height'], 
                    page.width, page.height
                )
                
                cells = self._extract_cells_with_plumber(page, pdf_bbox)
                
                if not cells:
                    logger.warning(f"No cells extracted for table {idx+1} on page {page_num}")
                    continue
                
                # Collect context
                global_idx = elem_index_map.get((page_num, tbl_elem['element_id']))
                context_before, context_after = [], []
                
                if global_idx is not None:
                    context_before, context_after = self._collect_context(
                        tbl_elem, all_elems, global_idx
                    )
                
                extracted_table = ExtractedTable(
                    page_number=page_num,
                    cells=[TableCell(**c) for c in cells],
                    bbox=tbl_elem['bbox'],
                    table_id=f"table_{idx+1:03d}",
                    reading_order=tbl_elem['reading_order'],
                    context_before=context_before,
                    context_after=context_after
                )
                
                extracted_tables.append(extracted_table)
                print(f"  Extracted table_{idx+1:03d} on page {page_num} with {len(cells)} cells, context: {len(context_before)} before, {len(context_after)} after")

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "extraction_config": {
                    "context_window_before": self.context_window_before,
                    "context_window_after": self.context_window_after
                },
                "total_tables": len(extracted_tables),
                "tables": [t.dict() for t in extracted_tables]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(extracted_tables)} tables with context to {output_path}")

if __name__ == "__main__":
    # Example usage
    raw_path = "raw_analysis.json"
    pdf_path = "document.pdf"
    output = "output_tables/tables.json"
    
    if os.path.exists(raw_path) and os.path.exists(pdf_path):
        extractor = TableExtractor(
            raw_path, 
            pdf_path,
            context_window_before=3,  # Configurable
            context_window_after=3    # Configurable
        )
        extractor.extract_and_save(output)
