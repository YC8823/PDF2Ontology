"""
Generic Text Extractor

Features:
1. Reads raw_analysis.json with flat structure.
2. Extracts text ordered by Reading Order.
3. Filters table regions (type='tab'/'table') to avoid duplication.
4. Intelligent paragraph merging.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TextBlock(BaseModel):
    page_number: int
    text: str

class TextExtractor:
    def __init__(self, raw_analysis_path: str):
        self.raw_analysis_path = raw_analysis_path
        with open(raw_analysis_path, 'r', encoding='utf-8') as f:
            self.raw_analysis = json.load(f)
            
    def extract_text(self) -> List[TextBlock]:
        text_blocks = []
        results = self.raw_analysis.get('results_per_page', [])
        
        # Types to exclude
        exclude_types = ['tab', 'table', 'figure', 'fig', 'image'] 
        
        for page_data in results:
            page_num = page_data.get('page_number', 1)
            elements = page_data.get('elements', [])
            
            # 1. Filter & Collect
            valid_elems = []
            for elem in elements:
                # Support both label and type keys
                label = elem.get('label', elem.get('type', 'text')).lower()
                text = elem.get('text', '').strip()
                
                if not text: continue
                if label in exclude_types: continue # Skip tables and images
                
                valid_elems.append({
                    'text': text,
                    'type': label,
                    'ro': elem.get('reading_order', 999)
                })
            
            # 2. Sort
            valid_elems.sort(key=lambda x: x['ro'])
            
            # 3. Merging Logic
            text_parts = []
            for e in valid_elems:
                t = e['text']
                l = e['type']
                
                # Add newlines based on type
                if l in ['header', 'sec_1']:
                    text_parts.append(f"\n\n## {t}\n")
                elif l in ['list']:
                    text_parts.append(f"\n- {t}")
                else:
                    text_parts.append(f"{t}\n")
            
            full_text = "".join(text_parts).strip()
            
            text_blocks.append(TextBlock(
                page_number=page_num,
                text=full_text
            ))
            
        return text_blocks

    def save(self, output_path: str):
        blocks = self.extract_text()
        data = {
            "document": self.raw_analysis.get('pdf_filename', 'unknown'),
            "pages": [b.dict() for b in blocks]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved text to {output_path}")

if __name__ == "__main__":
    raw_path = "raw_analysis.json"
    output = "output_text/text.json"
    
    if os.path.exists(raw_path):
        extractor = TextExtractor(raw_path)
        extractor.save(output)