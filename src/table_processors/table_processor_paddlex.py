import cv2
import numpy as np
from typing import List, Any, Dict, Optional

# Third-party libraries for model inference
from paddlex import create_pipeline
from paddleocr import PaddleOCR

# Core data structures and utility functions
from ..core.models import BoundingBox, RawTable, StructuredTable, TableCell
from ..utils.image_utils import preprocess_image_for_ocr, crop_image_by_bbox

# --- Concrete Model Runner Implementations ---

class PaddleTableRecognizer:
    """
    A wrapper for the PaddleX Table Recognition V2 pipeline.
    This class handles both table detection and structure recognition.
    """
    def __init__(self, device: str = "gpu:0", table_type: str = "wired"):
        try:
            # 明确指定工业级模型
            self.pipeline = create_pipeline(
                "table_recognition_v2",
                device=device,
                table_type=table_type,
                det_model_dir="en_PP-OCRv4_det_server",
                rec_model_dir="en_PP-OCRv4_rec_server",
                table_model_dir="en_ppstructure_server_v2.0_table_structure"
            )
            print("INFO: Using industrial-grade table recognition models")
        except Exception as e:
            print(f"ERROR: {e}")
            # 回退到默认模型
            self.pipeline = create_pipeline(
                "table_recognition_v2",
                device=device,
                table_type=table_type
            )

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs the full table recognition pipeline on an image.

        Args:
            image: A preprocessed image as a NumPy array.

        Returns:
            A list of result dictionaries, where each dictionary contains
            the 'bbox' and 'res' (structure and cells) for a detected table.
        """
        print("INFO: Starting table detection and structure recognition...")
        results = []
        # 遍历生成器结果
        for table_result in self.pipeline.predict(image):
            # 保存表格调试图像
            if hasattr(table_result, 'bbox'):
                x1, y1, x2, y2 = map(int, table_result.bbox)
                table_img = image[y1:y2, x1:x2]
                cv2.imwrite(f"debug_table_{len(results)}.jpg", table_img)
            
            # 尝试获取结构化数据
            html_content = table_result.html if hasattr(table_result, 'html') else ""
            json_content = table_result.json if hasattr(table_result, 'json') else {}
            
            # 从JSON内容提取单元格
            cells = []
            if json_content and 'cells' in json_content:
                cells = json_content['cells']
            # elif html_content:
            #     # 备选方案：从HTML解析表格结构
            #     cells = self._parse_cells_from_html(html_content)
            
            # 构建结果字典
            result_dict = {
                'bbox': table_result.bbox.tolist() if hasattr(table_result, 'bbox') else [],
                'res': {
                    'html': html_content,
                    'cells': cells
                }
            }
            
            # 添加详细调试信息
            print(f"DEBUG: Table {len(results)} - Cells found: {len(cells)}")
            print(f"DEBUG: HTML snippet: {html_content[:100]}...")
            
            results.append(result_dict)
        
        print(f"INFO: Found {len(results)} tables.")
        return results
    
    # def _parse_cells_from_html(self, html: str) -> List[Dict]:
    #     """从HTML表格结构解析单元格信息（备选方案）"""
    #     from bs4 import BeautifulSoup
    #     try:
    #         soup = BeautifulSoup(html, 'html.parser')
    #         table = soup.find('table')
    #         if not table:
    #             return []
                
    #         cells = []
    #         for row_idx, row in enumerate(table.find_all('tr')):
    #             for col_idx, cell in enumerate(row.find_all(['td', 'th'])):
    #                 # 计算跨行跨列
    #                 rowspan = int(cell.get('rowspan', 1))
    #                 colspan = int(cell.get('colspan', 1))
                    
    #                 cells.append({
    #                     'text': cell.get_text(strip=True),
    #                     'start_row': row_idx,
    #                     'end_row': row_idx + rowspan - 1,
    #                     'start_col': col_idx,
    #                     'end_col': col_idx + colspan - 1
    #                 })
    #         return cells
    #     except Exception as e:
    #         print(f"WARNING: HTML parsing failed: {e}")
    #         return []


class PaddleOcrEngine:
    """
    A wrapper for the PaddleOCR engine, configured for recognizing
    text within table cells, including custom symbols.
    """
    def __init__(self, use_gpu: bool = True, custom_dict_path: Optional[str] = None):
        print(f"INFO: Initializing PaddleOcrEngine (GPU: {use_gpu}).")
        # Define standard industrial symbols. In a real app, this could come from a config file.
        self.custom_symbols = {
            "CE_MARK": "✓",
            "DELTA": "Δ",
            "ROUGHNESS_Ra": "Ra"
        }
        try:
            self.ocr_enginec = PaddleOCR(
                # rec_model_name='PP-OCRv4_server_rec_doc',
                lang='en', 
                # rec_model_dir='./models/german_PP-OCRv4_rec_train',
                # det_model_dir='./models/german_PP-OCRv4_det_train',
                #rec_char_dict_path=custom_dict_path,
                #det_db_score_mode="slow", # Use high-precision mode for cell content
                # use_gpu=use_gpu,
                #show_log=False # Suppress noisy OCR logs
            )
            print("INFO: PaddleOcrEngine initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize PaddleOCR: {e}")
            raise

    def predict(self, cell_image: np.ndarray) -> str:
        """
        Performs OCR on a single cell image.

        Args:
            cell_image: An image of a table cell as a NumPy array.

        Returns:
            The recognized text content.
        """
        result = self.ocr_engine.predict(cell_image, cls=False)
        if result and result[0]:
            # The result is a list of lines, for a cell we expect one line
            text = "".join([line[1][0] for line in result[0]])
            return text
        return ""

    def standardize_symbols(self, text: str) -> str:
        """
        Replaces recognized special characters with standardized codes.
        """
        for code, char in self.custom_symbols.items():
            if char in text:
                text = text.replace(char, f"[{code}]")
        return text


# --- Main Processor Class ---

class TableProcessor:
    """
    A comprehensive processor for detecting, analyzing, and structuring tables
    from industrial document images using PaddlePaddle models.
    """
    def __init__(self, table_recognizer: PaddleTableRecognizer, ocr_engine: PaddleOcrEngine):
        """
        Initializes the processor with the necessary model runners.
        """
        self.table_recognizer = table_recognizer
        self.ocr_engine = ocr_engine

    def process(self, image: np.ndarray, document_id: str) -> List[StructuredTable]:
        """
        Main workflow to process a full-page image and extract structured tables.
        """
        print(f"INFO: Starting processing for document '{document_id}'.")
        preprocessed_image = preprocess_image_for_ocr(image)

        table_results = self.table_recognizer.predict(preprocessed_image)
        if not table_results:
            print("INFO: No tables found in the image.")
            return []

        structured_tables = []
        for i, table_res in enumerate(table_results):
            
            print(f"DEBUG: Table {i} keys: {table_res.keys()}")
            if 'bbox' not in table_res:
                print(f"WARNING: Table {i} missing 'bbox' key. Available keys: {table_res.keys()}")
                continue
            table_id = f"{document_id}_table_{i+1}"
            print(f"INFO: Processing {table_id}...")
            raw_structure = table_res
            
            # --- DEBUGGING: Print all available keys from the model's output ---
            print(f"DEBUG: Keys in table result: {raw_structure.keys()}")
            bbox_coords = None
            possible_bbox_keys = ['bbox', 'box', 'location', 'coords']
            for key in possible_bbox_keys:
                if key in raw_structure:
                    bbox_coords = raw_structure.get(key)
                    print(f"INFO: Found bounding box under the key: '{key}'")
                    break

            if bbox_coords is None:
                print(f"WARNING: Table result {i+1} is missing a recognized bounding box key. Skipping.")
                continue

            table_bbox = BoundingBox(x1=bbox_coords[0], y1=bbox_coords[1],
                                     x2=bbox_coords[2], y2=bbox_coords[3])

            # Pass the entire table result dictionary, which now contains the structure info.
            raw_table = self._process_individual_table(preprocessed_image, raw_structure, table_bbox)

            structured_table = self._format_to_structured_data(raw_table, table_id)
            structured_tables.append(structured_table)

        return structured_tables

    def _process_individual_table(self, image: np.ndarray, table_structure: Dict, bbox: BoundingBox) -> RawTable:
        """
            Processes a single detected table.
        """
        table_cells: List[TableCell] = []
        
        # 确保有单元格数据
        if not table_structure.get('cells'):
            print("WARNING: Table structure has no cells data")
            return RawTable(bbox=bbox, cells=[], structure_tokens=table_structure.get('html', ""))
        
        # 处理每个单元格
        for cell_info in table_structure['cells']:
            # 提取边界框坐标
            cell_bbox_coords = cell_info['bbox']
            
            # 裁剪单元格图像
            cell_image = crop_image_by_bbox(image, cell_bbox_coords)
            
            # OCR识别
            cell_text = self.ocr_engine.predict(cell_image)
            
            # 特殊内容处理
            if self._is_special_content(cell_text):
                cell_text = self.ocr_engine.standardize_symbols(cell_text)
            
            # 创建TableCell对象
            table_cells.append(
                TableCell(
                    bbox=BoundingBox(
                        x1=cell_bbox_coords[0], 
                        y1=cell_bbox_coords[1],
                        x2=cell_bbox_coords[2], 
                        y2=cell_bbox_coords[3]
                    ),
                    row_index=cell_info.get('start_row', -1),
                    col_index=cell_info.get('start_col', -1),
                    row_span=cell_info.get('end_row', 0) - cell_info.get('start_row', 0) + 1,
                    col_span=cell_info.get('end_col', 0) - cell_info.get('start_col', 0) + 1,
                    text=cell_text
                )
            )
        
        return RawTable(
            bbox=bbox, 
            cells=table_cells, 
            structure_tokens=table_structure.get('html', "")
        )
    
    def _is_special_content(self, text: str) -> bool:
        """
        Determines if a cell's text contains content that needs special handling.
        This is a placeholder for more advanced logic, e.g., checking against a
        list of keywords or using a classifier.
        """
        # Example check from reference code
        return "conformity" in text.lower() or any(char in text for char in self.ocr_engine.custom_symbols.values())

    def _format_to_structured_data(self, raw_table: RawTable, table_id: str) -> StructuredTable:
        """
        Converts the raw cell list into a structured format with headers and rows.
        This version correctly handles tables by mapping cells to a grid.
        """
        print(f"INFO: Formatting table {table_id} into a structured format.")
        if not raw_table.cells:
            return StructuredTable(table_id=table_id, headers=[], rows=[])

        max_row = max(c.row_index + c.row_span for c in raw_table.cells)
        max_col = max(c.col_index + c.col_span for c in raw_table.cells)
        
        # Create a grid to handle merged cells correctly
        grid = [[None for _ in range(max_col)] for _ in range(max_row)]

        # Populate the grid with cell text
        for cell in raw_table.cells:
            for r in range(cell.row_span):
                for c in range(cell.col_span):
                    if (cell.row_index + r < max_row) and (cell.col_index + c < max_col):
                        grid[cell.row_index + r][cell.col_index + c] = cell.text
        
        # Assume the first non-empty row is the header
        header_row_index = -1
        for i, row in enumerate(grid):
            if any(cell is not None for cell in row):
                header_row_index = i
                break

        if header_row_index == -1:
             return StructuredTable(table_id=table_id, headers=[], rows=[], metadata={"parsing_warnings": "No header row found."})

        headers = [str(h) if h is not None else "" for h in grid[header_row_index]]
        
        final_rows = []
        for i in range(header_row_index + 1, max_row):
            row_data = grid[i]
            if any(cell is not None for cell in row_data):
                row_dict = {headers[j]: (str(cell) if cell is not None else "") for j, cell in enumerate(row_data)}
                final_rows.append(row_dict)

        return StructuredTable(table_id=table_id, headers=headers, rows=final_rows)


if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Initializing Table Processing Pipeline ---")
    # In a real app, these would be instantiated once and reused.
    table_recognizer = PaddleTableRecognizer(device="gpu:0")
    # Point to the path where you have your custom dictionary file for OCR
    ocr_engine = PaddleOcrEngine(use_gpu=True, custom_dict_path=None) # e.g., 'custom_symbols.txt'

    # Create the main processor instance
    table_processor = TableProcessor(
        table_recognizer=table_recognizer,
        ocr_engine=ocr_engine
    )

    print("\n--- Processing a Document Image ---")
    # This requires a valid image path. A dummy image won't work with real models.

    image_path = "samples/page_3.png"
    try:
        # Load the image using OpenCV
        input_image = cv2.imread(image_path)
        if input_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Process the image
        structured_results = table_processor.process(image=input_image, document_id="report_001")

        print("\n--- Final Structured Output ---")
        if structured_results:
            for table in structured_results:
                print(table.model_dump_json(indent=2))
        else:
            print("No structured tables were extracted.")

    except (FileNotFoundError, Exception) as e:
        print(f"ERROR: An error occurred during processing: {e}")
        print("NOTE: Please ensure the image path is correct and all PaddlePaddle dependencies are installed.")