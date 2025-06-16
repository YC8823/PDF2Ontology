from typing import List, Dict, Any, Optional
from ..pydantic_models.table_models import TableStructure, TableCell, TableData
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import json

class TableContentExtractor:
    """Table content extractor using GPT-4o"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.1
        )
    
    def extract_table_content(self, image_path: str, bbox: Dict[str, float],
                            structure_info: Dict[str, Any]) -> TableData:
        """Extract table content based on structure information"""
        # Encode image
        base64_image = self._encode_image(image_path)
        
        # Create extraction prompt
        prompt = self._create_extraction_prompt(structure_info)
        
        # Call GPT-4o to extract content
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
        
        response = self.llm.invoke([message])
        
        # Parse response
        try:
            table_json = json.loads(response.content)
            return self._create_table_data(table_json, structure_info, bbox)
        except json.JSONDecodeError:
            # If JSON parsing fails, use fallback method
            return self._extract_with_fallback(response.content, structure_info, bbox)
    
    def _create_extraction_prompt(self, structure_info: Dict[str, Any]) -> str:
        """Create table content extraction prompt"""
        num_rows = structure_info["grid_structure"]["num_rows"]
        num_cols = structure_info["grid_structure"]["num_cols"]
        
        return f"""
Please extract the detailed content of the table in the image. Based on detected structure information:
- Estimated rows: {num_rows}
- Estimated columns: {num_cols}

Please output table content in the following JSON format:
{{
    "headers": ["Column 1 Title", "Column 2 Title", ...],
    "rows": [
        ["Row 1 Col 1", "Row 1 Col 2", ...],
        ["Row 2 Col 1", "Row 2 Col 2", ...],
        ...
    ],
    "caption": "Table title (if any)",
    "notes": "Table annotations or notes (if any)",
    "merged_cells": [
        {{"row": 1, "col": 2, "rowspan": 2, "colspan": 1, "content": "Merged cell content"}}
    ]
}}

Requirements:
1. Accurately identify headers and data rows
2. Preserve original formatting of cell content
3. Identify merged cells and mark their span information
4. Extract table title or notes if present
5. Ensure data completeness and accuracy
"""
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_table_data(self, table_json: Dict[str, Any], 
                          structure_info: Dict[str, Any],
                          bbox: Dict[str, float]) -> TableData:
        """Create TableData object from extracted JSON data"""
        headers = table_json.get("headers", [])
        rows_data = table_json.get("rows", [])
        merged_cells = table_json.get("merged_cells", [])
        
        # Create cell list
        cells = []
        
        # Add header cells
        for col, header in enumerate(headers):
            cell = TableCell(
                row=0,
                col=col,
                content=header,
                cell_type="header",
                confidence=0.9
            )
            cells.append(cell)
        
        # Add data cells
        for row_idx, row_data in enumerate(rows_data, start=1):
            for col_idx, cell_content in enumerate(row_data):
                cell = TableCell(
                    row=row_idx,
                    col=col_idx,
                    content=str(cell_content),
                    cell_type="data",
                    confidence=0.85
                )
                cells.append(cell)
        
        # Handle merged cells
        for merged_cell in merged_cells:
            # Update corresponding cell span information
            for cell in cells:
                if (cell.row == merged_cell["row"] and 
                    cell.col == merged_cell["col"]):
                    cell.rowspan = merged_cell.get("rowspan", 1)
                    cell.colspan = merged_cell.get("colspan", 1)
                    cell.content = merged_cell["content"]
                    break
        
        # Create table structure
        structure = TableStructure(
            rows=len(rows_data) + 1,  # +1 for header
            cols=len(headers) if headers else (max(len(row) for row in rows_data) if rows_data else 0),
            cells=cells,
            headers=headers,
            caption=table_json.get("caption"),
            page_number=1,  # Need to pass from external
            extraction_method="gpt4o_with_structure",
            structure_confidence=0.85
        )
        
        # Create data rows
        data_rows = []
        for row_data in rows_data:
            if headers:
                row_dict = dict(zip(headers, row_data))
            else:
                row_dict = {f"col_{i}": val for i, val in enumerate(row_data)}
            data_rows.append(row_dict)
        
        return TableData(
            structure=structure,
            data_rows=data_rows,
            metadata={
                "extraction_info": structure_info,
                "bbox": bbox,
                "notes": table_json.get("notes", "")
            }
        )
    
    def _extract_with_fallback(self, response_text: str, 
                              structure_info: Dict[str, Any],
                              bbox: Dict[str, float]) -> TableData:
        """Fallback extraction method using text parsing"""
        # Simple text parsing as fallback
        lines = response_text.strip().split('\n')
        headers = []
        rows_data = []
        
        for line in lines:
            if line.strip():
                # Simple splitting logic
                parts = [part.strip() for part in line.split('|') if part.strip()]
                if not headers and parts:
                    headers = parts
                elif parts:
                    rows_data.append(parts)
        
        # Create basic table data
        cells = []
        for col, header in enumerate(headers):
            cells.append(TableCell(
                row=0, col=col, content=header, 
                cell_type="header", confidence=0.7
            ))
        
        for row_idx, row_data in enumerate(rows_data, start=1):
            for col_idx, cell_content in enumerate(row_data):
                cells.append(TableCell(
                    row=row_idx, col=col_idx, content=cell_content,
                    cell_type="data", confidence=0.7
                ))
        
        structure = TableStructure(
            rows=len(rows_data) + 1,
            cols=len(headers),
            cells=cells,
            headers=headers,
            page_number=1,
            extraction_method="fallback_text_parsing",
            structure_confidence=0.7
        )
        
        data_rows = [dict(zip(headers, row)) for row in rows_data]
        
        return TableData(
            structure=structure,
            data_rows=data_rows,
            metadata={"extraction_info": structure_info, "bbox": bbox}
        )