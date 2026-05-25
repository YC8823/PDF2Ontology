# src/utils/text_table_preprocessor.py
"""
Simplified Text & Table Preprocessor

Core responsibilities:
1. Extract plain text from PDF pages
2. Extract tables with merged cell information
3. Associate tables with their titles
4. Output structured JSON

Removed features:
- Keyword-based filtering
- Context extraction strategies
"""

import logging
import os
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import pdfplumber
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ==================================================================
# === PYDANTIC MODELS
# ==================================================================

class TableCell(BaseModel):
    """Single table cell with merged cell information"""
    row: int = Field(description="Row index (0-based)")
    col: int = Field(description="Column index (0-based)")
    text: str = Field(description="Cell content text")
    rowspan: int = Field(description="Number of rows this cell spans", default=1)
    colspan: int = Field(description="Number of columns this cell spans", default=1)


class TableBlock(BaseModel):
    """Represents a table with metadata"""
    page_number: int = Field(description="Page number where table appears")
    title: Optional[str] = Field(description="Table title (if found)", default=None)
    cells: List[TableCell] = Field(description="List of table cells with merged cell info")
    bbox: Tuple[float, float, float, float] = Field(description="Bounding box (x0, y0, x1, y1)")
    table_id: str = Field(description="Unique table identifier")


class TextBlock(BaseModel):
    """Represents extracted plain text from a page"""
    page_number: int = Field(description="Page number")
    text: str = Field(description="Extracted plain text content")
    bbox: Optional[Tuple[float, float, float, float]] = Field(description="Text region bbox", default=None)


class ProcessedDocument(BaseModel):
    """Complete processed document structure"""
    document_name: str = Field(description="PDF filename")
    tables: List[TableBlock] = Field(description="All extracted tables")
    plain_text_blocks: List[TextBlock] = Field(description="Plain text from each page")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


# ==================================================================
# === CORE PROCESSOR
# ==================================================================

class TextTablePreprocessor:
    """
    Simplified PDF processor for text and table extraction.
    
    Key features:
    - Extract plain text from pages
    - Extract tables with merged cell information
    - Associate tables with titles using spatial proximity
    - No keyword filtering or complex context strategies
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Initialized TextTablePreprocessor (output: {output_dir})")
    
    # ========================================
    # MERGED CELL PROCESSING (from merged_cell_processor.py)
    # ========================================
    
    def _diagnose_raw_table(self, raw_table: List[List[Any]], table_index: int = 0):
        """
        Diagnostic function to print raw table structure for debugging.
        
        Args:
            raw_table: Raw table from pdfplumber
            table_index: Index of table for logging
        """
        logger.debug(f"\n{'='*60}")
        logger.debug(f"RAW TABLE DIAGNOSIS - Table {table_index}")
        logger.debug(f"{'='*60}")
        logger.debug(f"Dimensions: {len(raw_table)} rows x {len(raw_table[0]) if raw_table else 0} cols")
        
        for r_idx, row in enumerate(raw_table):
            row_repr = []
            for c_idx, cell in enumerate(row):
                if cell is None:
                    row_repr.append("[None]")
                elif isinstance(cell, str):
                    if cell.strip() == "":
                        row_repr.append("[Empty]")
                    else:
                        # Show first 15 chars
                        cell_preview = cell.replace('\n', '\\n')[:15]
                        row_repr.append(f"'{cell_preview}'")
                else:
                    row_repr.append(f"[{type(cell).__name__}]")
            
            logger.debug(f"Row {r_idx}: {' | '.join(row_repr)}")
        logger.debug(f"{'='*60}\n")
    
    def _detect_and_fix_merged_columns(
        self,
        cells: List[Dict[str, Any]],
        expected_col_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect and fix incorrectly merged columns in table cells.
        
        Common pattern: Multiple numeric values separated by spaces in a single cell
        Example: "9 10.5" should be split into "9" and "10.5"
        
        Args:
            cells: List of cell dictionaries
            expected_col_count: Expected number of columns (optional)
            
        Returns:
            Fixed list of cells
        """
        if not cells:
            return cells
        
        # Determine expected column count from header or first few rows
        if expected_col_count is None:
            # Find the row with most columns (likely the header)
            max_cols = max(cell['col'] + cell['colspan'] for cell in cells)
            expected_col_count = max_cols
        
        fixed_cells = []
        cells_by_row = {}
        
        # Group cells by row
        for cell in cells:
            row = cell['row']
            if row not in cells_by_row:
                cells_by_row[row] = []
            cells_by_row[row].append(cell)
        
        # Process each row
        for row_num in sorted(cells_by_row.keys()):
            row_cells = sorted(cells_by_row[row_num], key=lambda c: c['col'])
            
            # Check if this row has fewer columns than expected
            row_col_count = sum(cell['colspan'] for cell in row_cells)
            
            if row_col_count < expected_col_count:
                # Try to fix merged columns
                row_cells = self._fix_row_merged_columns(
                    row_cells, 
                    expected_col_count,
                    row_num
                )
            
            fixed_cells.extend(row_cells)
        
        return fixed_cells
    
    def _fix_row_merged_columns(
        self,
        row_cells: List[Dict[str, Any]],
        expected_col_count: int,
        row_num: int
    ) -> List[Dict[str, Any]]:
        """
        Fix merged columns in a single row.
        
        Strategy:
        1. Look for cells with multiple numeric values separated by spaces
        2. Split them into separate cells
        
        Args:
            row_cells: Cells in this row
            expected_col_count: Expected total columns
            row_num: Row number for logging
            
        Returns:
            Fixed row cells
        """
        fixed_row = []
        
        for cell in row_cells:
            text = cell['text'].strip()
            
            # Pattern: Multiple numbers separated by spaces
            # Examples: "9 10.5", "6 7.5", "100 120 140"
            if self._is_merged_numeric_cell(text):
                logger.debug(
                    f"Detected merged numeric cell at row {row_num}, col {cell['col']}: '{text}'"
                )
                
                # Split into separate values
                values = self._split_numeric_values(text)
                
                if len(values) > 1:
                    logger.info(
                        f"Splitting cell at row {row_num}, col {cell['col']}: "
                        f"'{text}' -> {values}"
                    )
                    
                    # Create separate cells for each value
                    for idx, value in enumerate(values):
                        new_cell = {
                            'row': cell['row'],
                            'col': cell['col'] + idx,  # Shift column index
                            'text': value,
                            'rowspan': cell['rowspan'],
                            'colspan': 1  # Each split cell has colspan=1
                        }
                        fixed_row.append(new_cell)
                else:
                    # Not actually merged, keep as is
                    fixed_row.append(cell)
            else:
                # Not a merged numeric cell, keep as is
                fixed_row.append(cell)
        
        return fixed_row
    
    def _is_merged_numeric_cell(self, text: str) -> bool:
        """
        Check if text looks like merged numeric values.
        
        Pattern: Two or more numbers separated by spaces
        Examples: "9 10.5", "100 120", "6 7.5"
        
        Args:
            text: Cell text
            
        Returns:
            True if it looks like a merged numeric cell
        """
        if not text or len(text) < 3:
            return False
        
        # Remove common non-numeric prefixes/suffixes
        text = text.strip()
        
        # Pattern: number [space] number
        # Allow decimals, negatives, and scientific notation
        import re
        pattern = r'^-?\d+\.?\d*\s+-?\d+\.?\d*'
        
        return bool(re.match(pattern, text))
    
    def _split_numeric_values(self, text: str) -> List[str]:
        """
        Split text containing multiple numeric values.
        
        Args:
            text: Text to split (e.g., "9 10.5")
            
        Returns:
            List of individual values
        """
        import re
        
        # Pattern to match numbers (including decimals)
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        
        return [m for m in matches if m]
    
    def _table_to_merged_cell_json(self, table: pdfplumber.table.Table) -> List[Dict[str, Any]]:
        """
        Convert pdfplumber table to JSON with rowspan/colspan information.
        
        This method is adapted from merged_cell_processor.py with the same algorithm.
        
        Args:
            table: pdfplumber table object
            
        Returns:
            List of cell dictionaries with merged cell information
        """
        raw_table = table.extract()
        if not raw_table:
            return []
        
        # Enable diagnostic output for debugging (set to False in production)
        if logger.isEnabledFor(logging.DEBUG):
            self._diagnose_raw_table(raw_table)

        row_count = len(raw_table)
        col_count = len(raw_table[0]) if row_count > 0 else 0
        
        # Track visited cells
        visited = [[False for _ in range(col_count)] for _ in range(row_count)]
        cells = []

        for r in range(row_count):
            for c in range(col_count):
                if visited[r][c]:
                    continue

                cell_value = raw_table[r][c]
                
                # Helper function to check if a cell is "empty" (None or merged)
                def is_empty_cell(value):
                    """Check if cell is None or part of merged cell"""
                    return value is None
                
                # Calculate rowspan
                rowspan = 1
                while r + rowspan < row_count and is_empty_cell(raw_table[r + rowspan][c]):
                    rowspan += 1
                
                # Calculate colspan
                colspan = 1
                while c + colspan < col_count and is_empty_cell(raw_table[r][c + colspan]):
                    # Ensure entire merged region has None/empty values
                    is_merge_candidate = True
                    for i in range(rowspan):
                        if not is_empty_cell(raw_table[r + i][c + colspan]):
                            is_merge_candidate = False
                            break
                    if is_merge_candidate:
                        colspan += 1
                    else:
                        break

                # Mark all cells in merged region as visited
                for i in range(rowspan):
                    for j in range(colspan):
                        visited[r + i][c + j] = True
                
                # Convert cell text
                cell_text = str(cell_value or '').strip()
                
                cells.append({
                    "row": r,
                    "col": c,
                    "text": cell_text,
                    "rowspan": rowspan,
                    "colspan": colspan
                })
        
        return cells
    
    # ========================================
    # TABLE TITLE EXTRACTION
    # ========================================
    
    def _find_table_title(
        self, 
        page: pdfplumber.page.Page, 
        table_bbox: Tuple[float, float, float, float],
        search_distance: float = 25.0
    ) -> Optional[str]:
        """
        Find table title by searching above the table.
        
        Strategy:
        1. Define search region above table (within search_distance)
        2. Extract text from that region
        3. Clean and return title text
        
        Args:
            page: pdfplumber page object
            table_bbox: Table bounding box (x0, y0, x1, y1)
            search_distance: Maximum distance above table to search for title
            
        Returns:
            Table title or None
        """
        x0, y0, x1, y1 = table_bbox
        
        # Define search region above table
        title_bbox = (
            x0,
            max(0, y0 - search_distance),
            x1,
            y0
        )
        
        # Validate bbox
        if title_bbox[3] <= title_bbox[1]:
            return None
        
        try:
            # Crop and extract text
            title_crop = page.crop(title_bbox)
            title_text = title_crop.extract_text(x_tolerance=2, y_tolerance=2)
            
            if title_text:
                # Clean title text
                title_text = self._clean_title_text(title_text)
                if title_text:
                    logger.debug(f"Found table title: '{title_text}'")
                    return title_text
        except Exception as e:
            logger.debug(f"Could not extract title: {e}")
        
        return None
    
    def _clean_title_text(self, text: str) -> str:
        """
        Clean extracted title text.
        
        Rules:
        - Remove excessive whitespace
        - Remove line breaks
        - Trim to reasonable length
        
        Args:
            text: Raw title text
            
        Returns:
            Cleaned title text
        """
        if not text:
            return ""
        
        # Replace line breaks with spaces
        text = text.replace('\n', ' ')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        # Limit length (table titles should be concise)
        if len(text) > 200:
            text = text[:200] + "..."
        
        return text
    
    # ========================================
    # PLAIN TEXT EXTRACTION
    # ========================================
    
    # def _extract_page_text(
    #     self, 
    #     page: pdfplumber.page.Page, 
    #     page_number: int
    # ) -> TextBlock:
    #     """
    #     Extract plain text from a page.
        
    #     Args:
    #         page: pdfplumber page object
    #         page_number: Page number (1-based)
            
    #     Returns:
    #         TextBlock with extracted text
    #     """
    #     try:
    #         text = page.extract_text(x_tolerance=2, y_tolerance=2)
    #         if not text:
    #             text = ""
            
    #         return TextBlock(
    #             page_number=page_number,
    #             text=text.strip(),
    #             bbox=None  # Could add page bbox if needed
    #         )
    #     except Exception as e:
    #         logger.error(f"Error extracting text from page {page_number}: {e}")
    #         return TextBlock(
    #             page_number=page_number,
    #             text="",
    #             bbox=None
    #         )
    
    # ========================================
    # TABLE EXTRACTION
    # ========================================
    
    def _extract_page_tables(
        self,
        page: pdfplumber.page.Page,
        page_number: int,
        base_table_id: int
    ) -> Tuple[List[TableBlock], int]:
        """
        Extract all tables from a page.
        
        Args:
            page: pdfplumber page object
            page_number: Page number (1-based)
            base_table_id: Starting table ID for this page
            
        Returns:
            Tuple of (list of TableBlocks, next_table_id)
        """
        tables = []
        table_id = base_table_id
        
        pdfplumber_tables = page.find_tables()
        
        for table in pdfplumber_tables:
            # Extract title
            title = self._find_table_title(page, table.bbox)
            
            # Convert to merged cell format
            cells_data = self._table_to_merged_cell_json(table)
            
            if not cells_data:
                logger.warning(f"Table on page {page_number} has no cells, skipping")
                continue
            
            # Apply fix for incorrectly merged columns
            logger.debug(f"Checking for merged columns in table on page {page_number}...")
            cells_data = self._detect_and_fix_merged_columns(cells_data)
            
            # Convert to Pydantic models
            cells = [TableCell(**cell) for cell in cells_data]
            
            # Create table block
            table_block = TableBlock(
                page_number=page_number,
                title=title,
                cells=cells,
                bbox=table.bbox,
                table_id=f"table_{table_id:03d}"
            )
            
            tables.append(table_block)
            table_id += 1
            
            logger.debug(
                f"Extracted table_{table_id-1:03d} from page {page_number}: "
                f"{len(cells)} cells, title='{title}'"
            )
        
        return tables, table_id
    
    # ========================================
    # CROSS-PAGE TABLE MERGING
    # ========================================
    
    def _merge_cross_page_tables(self, tables: List[TableBlock]) -> List[TableBlock]:
        """
        Merge tables that span across consecutive pages.
        
        Criteria for merging:
        - Tables on consecutive pages
        - Same number of columns
        - Similar horizontal alignment
        
        Args:
            tables: List of table blocks sorted by page and position
            
        Returns:
            List of tables with cross-page tables merged
        """
        if len(tables) < 2:
            return tables
        
        merged_tables = []
        i = 0
        
        while i < len(tables):
            current_table = tables[i]
            
            if i + 1 < len(tables):
                next_table = tables[i + 1]
                
                # Check if tables are on consecutive pages
                if next_table.page_number == current_table.page_number + 1:
                    # Check if they have same number of columns
                    current_cols = max(cell.col + cell.colspan for cell in current_table.cells)
                    next_cols = max(cell.col + cell.colspan for cell in next_table.cells)
                    
                    if current_cols == next_cols and current_cols > 0:
                        # Merge tables
                        logger.info(
                            f"Merging cross-page tables: {current_table.table_id} "
                            f"(page {current_table.page_number}) + {next_table.table_id} "
                            f"(page {next_table.page_number})"
                        )
                        
                        # Calculate row offset
                        current_max_row = max(cell.row + cell.rowspan for cell in current_table.cells)
                        
                        # Adjust next table's row indices
                        adjusted_cells = []
                        for cell in next_table.cells:
                            adjusted_cell = TableCell(
                                row=cell.row + current_max_row,
                                col=cell.col,
                                text=cell.text,
                                rowspan=cell.rowspan,
                                colspan=cell.colspan
                            )
                            adjusted_cells.append(adjusted_cell)
                        
                        # Merge cells
                        current_table.cells.extend(adjusted_cells)
                        
                        # Skip next table
                        i += 1
            
            merged_tables.append(current_table)
            i += 1
        
        return merged_tables
    
     # ========================================
    # PLAIN TEXT EXTRACTION (IMPROVED)
    # ========================================
    
    def _extract_page_text(
        self, 
        page: pdfplumber.page.Page, 
        page_number: int,
        exclude_regions: List[Tuple[float, float, float, float]] = None
    ) -> TextBlock:
        """
        Extract plain text from a page, excluding specified regions.
        
        Strategy:
        1. Get all text objects from page
        2. Filter out text within exclude_regions (tables, images)
        3. Return only "pure" text content
        
        Args:
            page: pdfplumber page object
            page_number: Page number (1-based)
            exclude_regions: List of bboxes to exclude (x0, y0, x1, y1)
            
        Returns:
            TextBlock with extracted text (excluding tables/images)
        """
        if exclude_regions is None:
            exclude_regions = []
        
        try:
            # Extract all words with their bounding boxes
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False
            )
            
            if not words:
                return TextBlock(
                    page_number=page_number,
                    text="",
                    bbox=None
                )
            
            # Filter words that are NOT in exclude regions
            filtered_words = []
            for word in words:
                word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
                
                # Check if word overlaps with any exclude region
                if not self._is_bbox_in_exclude_regions(word_bbox, exclude_regions):
                    filtered_words.append(word)
            
            # Reconstruct text from filtered words
            # Sort by vertical position, then horizontal
            filtered_words.sort(key=lambda w: (w['top'], w['x0']))
            
            # Group words into lines
            lines = self._group_words_into_lines(filtered_words)
            
            # Join lines
            text = '\n'.join(lines)
            
            logger.debug(
                f"Page {page_number}: Extracted {len(filtered_words)}/{len(words)} words "
                f"({len(exclude_regions)} regions excluded)"
            )
            
            return TextBlock(
                page_number=page_number,
                text=text.strip(),
                bbox=None
            )
            
        except Exception as e:
            logger.error(f"Error extracting text from page {page_number}: {e}")
            return TextBlock(
                page_number=page_number,
                text="",
                bbox=None
            )
    
    def _is_bbox_in_exclude_regions(
        self,
        bbox: Tuple[float, float, float, float],
        exclude_regions: List[Tuple[float, float, float, float]],
        overlap_threshold: float = 0.5
    ) -> bool:
        """
        Check if a bounding box overlaps with any exclude region.
        
        Args:
            bbox: Bounding box to check (x0, y0, x1, y1)
            exclude_regions: List of exclude bboxes
            overlap_threshold: Minimum overlap ratio to consider as "inside"
            
        Returns:
            True if bbox should be excluded
        """
        x0, y0, x1, y1 = bbox
        bbox_area = (x1 - x0) * (y1 - y0)
        
        if bbox_area <= 0:
            return False
        
        for ex0, ey0, ex1, ey1 in exclude_regions:
            # Calculate intersection
            ix0 = max(x0, ex0)
            iy0 = max(y0, ey0)
            ix1 = min(x1, ex1)
            iy1 = min(y1, ey1)
            
            if ix0 < ix1 and iy0 < iy1:
                intersection_area = (ix1 - ix0) * (iy1 - iy0)
                overlap_ratio = intersection_area / bbox_area
                
                if overlap_ratio >= overlap_threshold:
                    return True
        
        return False
    
    def _group_words_into_lines(
        self,
        words: List[Dict],
        y_tolerance: float = 3.0
    ) -> List[str]:
        """
        Group words into lines based on vertical position.
        
        Args:
            words: List of word dictionaries with 'text', 'top', 'x0'
            y_tolerance: Maximum vertical distance to consider same line
            
        Returns:
            List of line strings
        """
        if not words:
            return []
        
        lines = []
        current_line_words = [words[0]]
        current_y = words[0]['top']
        
        for word in words[1:]:
            # Check if word is on same line
            if abs(word['top'] - current_y) <= y_tolerance:
                current_line_words.append(word)
            else:
                # Start new line
                line_text = ' '.join(w['text'] for w in current_line_words)
                lines.append(line_text)
                
                current_line_words = [word]
                current_y = word['top']
        
        # Add last line
        if current_line_words:
            line_text = ' '.join(w['text'] for w in current_line_words)
            lines.append(line_text)
        
        return lines
    
    # ========================================
    # COLLECT EXCLUDE REGIONS
    # ========================================
    
    def _collect_exclude_regions(
        self,
        page: pdfplumber.page.Page,
        tables_on_page: List[TableBlock],
        title_search_distance: float = 50.0
    ) -> List[Tuple[float, float, float, float]]:
        """
        Collect all regions that should be excluded from text extraction.
        
        Includes:
        1. Table bboxes
        2. Table title regions (area above tables)
        3. Image/figure regions (if available)
        
        Args:
            page: pdfplumber page object
            tables_on_page: List of tables on this page
            title_search_distance: Distance above table to exclude for title
            
        Returns:
            List of exclude region bboxes
        """
        exclude_regions = []
        
        # Add table regions
        for table in tables_on_page:
            x0, y0, x1, y1 = table.bbox
            exclude_regions.append((x0, y0, x1, y1))
            
            # Add title region above table
            title_region = (
                x0,
                max(0, y0 - title_search_distance),
                x1,
                y0
            )
            exclude_regions.append(title_region)
        
        # Add image/figure regions (if pdfplumber can detect them)
        try:
            images = page.images
            for img in images:
                img_bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                exclude_regions.append(img_bbox)
                
                # Add caption region below image
                caption_region = (
                    img['x0'],
                    img['bottom'],
                    img['x1'],
                    min(page.height, img['bottom'] + title_search_distance)
                )
                exclude_regions.append(caption_region)
        except Exception as e:
            logger.debug(f"Could not extract image regions: {e}")
        
        return exclude_regions
    
    # ========================================
    # UPDATED MAIN PROCESSING PIPELINE
    # ========================================
    
    def process_document(self, pdf_path: str) -> ProcessedDocument:
        """
        Main processing pipeline.
        
        Steps:
        1. Extract tables with merged cell info (first pass)
        2. Extract plain text EXCLUDING table/image regions (second pass)
        3. Merge cross-page tables
        4. Return structured result
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ProcessedDocument with all extracted information
        """
        doc_name = os.path.basename(pdf_path)
        logger.info(f"Processing document: {doc_name}")
        
        all_tables = []
        all_text_blocks = []
        table_id_counter = 1
        
        # Store page objects for second pass
        page_objects = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # ===== FIRST PASS: Extract tables =====
                logger.info("First pass: Extracting tables...")
                for page_num, page in enumerate(pdf.pages, 1):
                    page_objects.append(page)
                    
                    page_tables, table_id_counter = self._extract_page_tables(
                        page, page_num, table_id_counter
                    )
                    all_tables.extend(page_tables)
                    
                    logger.debug(f"Page {page_num}/{total_pages}: Found {len(page_tables)} tables")
                
                # ===== SECOND PASS: Extract text (excluding table regions) =====
                logger.info("Second pass: Extracting plain text (excluding tables/images)...")
                for page_num, page in enumerate(page_objects, 1):
                    # Get tables on this page
                    tables_on_page = [t for t in all_tables if t.page_number == page_num]
                    
                    # Collect exclude regions
                    exclude_regions = self._collect_exclude_regions(
                        page, tables_on_page
                    )
                    
                    # Extract text with exclusions
                    text_block = self._extract_page_text(
                        page, page_num, exclude_regions
                    )
                    all_text_blocks.append(text_block)
        
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            raise
        
        # Merge cross-page tables
        logger.info("Merging cross-page tables...")
        merged_tables = self._merge_cross_page_tables(all_tables)
        
        # Create metadata
        metadata = {
            "total_pages": total_pages,
            "total_tables_found": len(all_tables),
            "tables_after_merge": len(merged_tables),
            "total_text_blocks": len(all_text_blocks),
            "total_text_length": sum(len(tb.text) for tb in all_text_blocks)
        }
        
        logger.info(f"Processing complete:")
        logger.info(f"  - Pages: {metadata['total_pages']}")
        logger.info(f"  - Tables found: {metadata['total_tables_found']}")
        logger.info(f"  - After merge: {metadata['tables_after_merge']}")
        logger.info(f"  - Text blocks: {metadata['total_text_blocks']}")
        logger.info(f"  - Total text length: {metadata['total_text_length']} chars")
        
        return ProcessedDocument(
            document_name=doc_name,
            tables=merged_tables,
            plain_text_blocks=all_text_blocks,
            metadata=metadata
        )
    
    # ==================================================================
# === EXAMPLE USAGE
# ==================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_pdf_path = os.path.join(project_root, "data", "inputs", "t58740en.pdf")
    output_dir = os.path.join(project_root, "data", "outputs", "simplified_preprocessing")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create preprocessor
    processor = TextTablePreprocessor(output_dir=output_dir)
    
    # Process document
    processed_doc = processor.process_document(test_pdf_path)
    
    # Save results
    output_json_path = os.path.join(output_dir, "processed_document.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        f.write(processed_doc.model_dump_json(indent=2))
    
    logger.info(f"Results saved to: {output_json_path}")