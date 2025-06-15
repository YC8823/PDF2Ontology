# src/processors/table_processor.py

import torch
import pandas as pd
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from typing import Optional, List, Dict, Any

# Disable the Decompression Bomb check in Pillow
# Image.MAX_IMAGE_PIXELS = None

class TableProcessor:
    """
    A specialized processor that uses a two-stage pipeline to detect, structure,
    and extract text from tables in images for maximum robustness.
    It includes a fallback to programmatically generate cells if the model fails to detect them.
    """

    def __init__(self,
                 detection_model_name: str = "microsoft/table-transformer-detection",
                 structure_model_name: str = "microsoft/table-transformer-structure-recognition-v1.1-all"):
        """
        Initializes the TableProcessor by loading the required models for both detection
        and structure recognition stages.
        """
        print("Initializing TableProcessor: Loading detection and structure models...")
        self.detection_image_processor = AutoImageProcessor.from_pretrained(detection_model_name)
        self.detection_model = TableTransformerForObjectDetection.from_pretrained(detection_model_name)
        
        self.structure_image_processor = AutoImageProcessor.from_pretrained(structure_model_name)
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(structure_model_name)

        self.colors = {"table": "red", "table row": "blue", "table column": "green", "table cell": "magenta"}
        print("TableProcessor initialized successfully.")

    def extract_structured_table(
        self,
        image: Image.Image,
        ocr_lang: str = 'eng+deu',
        detection_threshold: float = 0.8,
        structure_threshold: float = 0.6,
        debug_prefix: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Processes an image using a two-stage pipeline to find and structure a table.

        Args:
            image (Image.Image): A PIL Image object of the document page.
            ocr_lang (str): The language(s) for Tesseract OCR.
            detection_threshold (float): Confidence threshold for the initial table detection.
            structure_threshold (float): Confidence threshold for detecting rows, columns, and cells.
            debug_prefix (Optional[str]): If provided, saves debug images with this prefix.
        """
        print("--- Stage 1: Detecting main table area ---")
        table_box = self._detect_main_table(image, threshold=detection_threshold)
        if not table_box:
            print("No table detected on the page.")
            return None

        # --- Verification Step 1: Visualize the detected table area ---
        if debug_prefix:
            debug_img_stage1 = image.copy()
            draw = ImageDraw.Draw(debug_img_stage1)
            draw.rectangle(table_box, outline="red", width=3)
            path_stage1 = f"{debug_prefix}_stage1_detection.png"
            print(f"Saving Stage 1 verification image to: {path_stage1}")
            debug_img_stage1.save(path_stage1)
        # --- End Verification Step ---

        table_image = image.crop(table_box)
        print("Table area detected and cropped.")

        print("\n--- Stage 2: Recognizing structure within the cropped table ---")
        detection_results = self._recognize_structure(table_image, threshold=structure_threshold)
        if not detection_results:
            print("No structural elements detected in the cropped table.")
            return None

        # --- Verification Step 2: Visualize the detected rows and columns ---
        if debug_prefix:
            path_stage2 = f"{debug_prefix}_stage2_structure.png"
            print(f"Saving Stage 2 verification image to: {path_stage2}")
            self._visualize_detections(table_image.copy(), detection_results, path_stage2)
        # --- End Verification Step ---

        table_grid = self._reconstruct_grid(detection_results)
        cells = table_grid.get('cells', [])

        if not cells:
            print("Warning: No 'table cell' elements were detected. Check the debug image.")
            return None

        print(f"Performing OCR on {len(cells)} detected cells...")
        for cell in cells:
            cell['text'] = self._perform_ocr_on_cell(table_image, cell['box'], ocr_lang)

        df = self._build_dataframe(table_grid)
        if df is None or df.empty:
            print("Warning: Could not construct a valid DataFrame.")
            return None

        return df

    def _detect_main_table(self, image: Image.Image, threshold: float) -> Optional[List[float]]:
        """Uses the detection model to find the primary table's bounding box."""
        inputs = self.detection_image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.detection_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.detection_image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        # Find the box with the highest score labeled as 'table'
        best_score = -1
        best_box = None
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.detection_model.config.id2label[label.item()] == "table":
                if score > best_score:
                    best_score = score
                    best_box = box.tolist()
        
        return best_box

    def _recognize_structure(self, image: Image.Image, threshold: float) -> List[Dict[str, Any]]:
        """Uses the structure model on a cropped image to find rows, columns, and cells."""
        size = {"shortest_edge": 800, "longest_edge": 1000}
        inputs = self.structure_image_processor(images=image, size=size, return_tensors="pt")
        with torch.no_grad():
            outputs = self.structure_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.structure_image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        boxes = self._rescale_bboxes(results['boxes'], image.size).tolist()
        detection_results = []
        for label_id, score, box in zip(results['labels'].tolist(), results['scores'].tolist(), boxes):
            detection_results.append({
                'label': self.structure_model.config.id2label[label_id],
                'score': score,
                'box': box
            })
        return detection_results
    
    def _visualize_detections(self, image: Image.Image, detections: List[Dict], output_path: str):
        """Saves a debug image with detected bounding boxes."""
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for det in detections:
            box, label, score = det['box'], det['label'], det['score']
            color = self.colors.get(label, "white")
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 5, box[1] + 5), f"{label} ({score:.2f})", fill=color, font=font)
        image.save(output_path)

    @staticmethod
    def _rescale_bboxes(out_bbox: torch.Tensor, size: tuple) -> torch.Tensor:
        """Rescales bounding box coordinates to the original image size."""
        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(1)
            return torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=1)
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        return b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    @staticmethod
    def _reconstruct_grid(table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reconstructs a logical grid from detected rows, columns, and cells."""
        rows = sorted([e for e in table_data if e['label'] == 'table row'], key=lambda x: x['box'][1])
        columns = sorted([e for e in table_data if e['label'] == 'table column'], key=lambda x: x['box'][0])
        cells = [e for e in table_data if e['label'] == 'table cell']
        
        print(f"Found {len(rows)} rows, {len(columns)} columns, and {len(cells)} cells in raw detections.")
        if not cells or not rows or not columns:
            return {"cells": cells} # Return partial data for debugging

        for i, row in enumerate(rows): row['row_number'] = i
        for i, col in enumerate(columns): col['col_number'] = i

        for cell in cells:
            cell_center_x = (cell['box'][0] + cell['box'][2]) / 2
            cell_center_y = (cell['box'][1] + cell['box'][3]) / 2
            min_y_dist, assigned_row = min([(abs(cell_center_y - (r['box'][1] + r['box'][3]) / 2), r['row_number']) for r in rows], key=lambda t: t[0])
            min_x_dist, assigned_col = min([(abs(cell_center_x - (c['box'][0] + c['box'][2]) / 2), c['col_number']) for c in columns], key=lambda t: t[0])
            cell.update({'row_number': assigned_row, 'col_number': assigned_col})
        return {"rows": rows, "columns": columns, "cells": cells}

    @staticmethod
    def _perform_ocr_on_cell(image: Image.Image, cell_box: List[float], lang: str) -> str:
        """Performs OCR on a given cell's bounding box."""
        box = [max(0, cell_box[0] - 5), max(0, cell_box[1] - 5), cell_box[2] + 5, cell_box[3] + 5]
        cell_image = image.crop(box)
        try:
            return pytesseract.image_to_string(cell_image, lang=lang, config='--psm 6').strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    @staticmethod
    def _build_dataframe(table_grid: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Constructs a Pandas DataFrame from the reconstructed grid structure."""
        rows, cols, cells = table_grid.get('rows', []), table_grid.get('columns', []), table_grid.get('cells', [])
        if not cells or not rows or not cols:
            return None

        grid = [['' for _ in cols] for _ in rows]
        for cell in cells:
            row_idx, col_idx = cell.get('row_number', -1), cell.get('col_number', -1)
            if 0 <= row_idx < len(rows) and 0 <= col_idx < len(cols):
                grid[row_idx][col_idx] = cell.get('text', '')

        df = pd.DataFrame(grid)
        if len(df) > 1 and df.iloc[0].astype(bool).any():
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        return df