# src/analyzers/dolphin_result_analyzer.py
"""
Dolphin Result Analyzer (Advanced)

Implements the advanced matching and archiving logic based on the
provided technical guide.

V2 Enhancements:
- Smarter caption matching: Uses spatial distance if two
  reading-order-adjacent candidates exist.
- Sequential IDs: Generates human-readable IDs
  (e.g., "page_3_image_1").
- Richer table info: Includes table ID, page, and content snippet
  in final archive.
"""

import os
import sys
import json
import argparse
import re
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: 'PyMuPDF' library not found.", file=sys.stderr)
    print("Please install: pip install PyMuPDF", file=sys.stderr)
    sys.exit(1)

# Default output subdirectory if none is provided
DEFAULT_OUTPUT_SUBDIR = "processed_archives"


def get_images_for_cropping(pdf_path: str, dpi: int = 300) -> Dict[int, Image.Image]:
    """
    Converts PDF to a dictionary map of {page_num: PIL.Image}
    for easy lookup during cropping.
    """
    print(f"Loading PDF for cropping (DPI: {dpi})...")
    images_map = {}
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            if pix.alpha:
                image = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
                image = image.convert("RGB")
            else:
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            images_map[page_num + 1] = image  # Page numbers start at 1
        
        doc.close()
        print(f"✓ PDF loaded, {len(images_map)} pages ready for cropping.")
        return images_map
        
    except Exception as e:
        print(f"Error: PDF conversion failed - {e}", file=sys.stderr)
        sys.exit(1)


def preprocess_raw_data(raw_data: Dict) -> Tuple[List, Dict, List, List, List]:
    """
    Phase 1: Data Preprocessing
    Loads raw data and builds the core data structures.
    """
    print("Phase 1: Preprocessing raw data...")
    all_blocks = []
    blocks_by_page = defaultdict(list)
    images = []
    tables = []
    paragraphs = [] # Includes all text-like elements

    text_types = ['para', 'header', 'list', 'cap', 'sec', 'text', 'foot']

    try:
        # Use 'results_per_page' from the simplified client
        results_per_page = raw_data['results_per_page']
        
        for page_result in results_per_page:
            if 'error' in page_result:
                continue
            
            page_num = page_result['page_number']
            
            # Use 'elements' and 'element_contents' from the new server output
            elements = page_result.get('elements', [])
            # 'element_contents' is now at the page level
            contents = page_result.get('element_contents', {}) 
            if not contents:
                # Fallback if server output is from demo_page.py (content inside element)
                for elem in elements:
                    if 'content' not in elem and 'text' in elem:
                         elem['content'] = elem['text']
                    contents[elem.get('element_id')] = elem.get('content')

            for elem in elements:
                # Inject content and page number into the element block
                elem['content'] = contents.get(elem['element_id'], '')
                elem['page_number'] = page_num
                
                # Use 'pixel_bbox' as the standard box
                if 'pixel_bbox' in elem:
                    elem['box'] = elem['pixel_bbox'] 
                
                all_blocks.append(elem)
                blocks_by_page[page_num].append(elem)
                
                elem_type = elem.get('type')
                if elem_type == 'fig':
                    images.append(elem)
                elif elem_type == 'tab':
                    tables.append(elem)
                elif elem_type in text_types:
                    paragraphs.append(elem)

        # Sort all blocks by page, then reading order
        all_blocks_sorted = sorted(all_blocks, key=lambda b: (b.get('page_number', 0), b.get('reading_order', 0)))
        
        print(f"✓ Preprocessing complete. Found:")
        print(f"  - {len(all_blocks_sorted)} total elements")
        print(f"  - {len(images)} images")
        print(f"  - {len(tables)} tables")
        print(f"  - {len(paragraphs)} text blocks")
        
        return all_blocks_sorted, blocks_by_page, images, tables, paragraphs

    except KeyError as e:
        print(f"Error: Input JSON missing expected key: {e}", file=sys.stderr)
        print("Please ensure you are using the _raw_analysis.json file.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during preprocessing: {e}", file=sys.stderr)
        sys.exit(1)

def get_vertical_distance(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculates the vertical (Y-axis) distance between two bboxes.
    Returns 0 if they are overlapping.
    """
    if not bbox1 or not bbox2:
        return float('inf')
        
    y1_bottom = bbox1[3]
    y2_top = bbox2[1]
    
    y1_top = bbox1[1]
    y2_bottom = bbox2[3]

    # Case 1: bbox2 is below bbox1
    dist_below = y2_top - y1_bottom
    if dist_below >= 0:
        return dist_below

    # Case 2: bbox2 is above bbox1
    dist_above = y1_top - y2_bottom
    if dist_above >= 0:
        return dist_above

    # Case 3: Bboxes are overlapping vertically
    return 0

def find_caption_for_image(
    image_block: Dict, 
    all_blocks_sorted: List[Dict],
    global_index: int,
    blocks_by_page: Dict[int, List[Dict]]
) -> Dict:
    """
    Phase 3.1: Find caption for an image.
    Uses globally sorted list index, same-page constraint,
    and spatial distance as a tie-breaker.
    """
    page_num = image_block['page_number']
    img_bbox = image_block.get('box') # Use 'box' which we standardized to 'pixel_bbox'
    caption_types = ['cap', 'header', 'text', 'para']
    
    candidate_after = None
    candidate_before = None

    # Strategy 1: Check element immediately after (RO + 1)
    if global_index + 1 < len(all_blocks_sorted):
        next_block = all_blocks_sorted[global_index + 1]
        if (next_block['page_number'] == page_num and 
            next_block['type'] in caption_types):
            candidate_after = next_block

    # Strategy 2: Check element immediately before (RO - 1)
    if global_index - 1 >= 0:
        prev_block = all_blocks_sorted[global_index - 1]
        if (prev_block['page_number'] == page_num and 
            prev_block['type'] in caption_types):
            candidate_before = prev_block

    chosen_block = None
    match_method = "N/A"
    base_confidence = 0.0

    # --- Adjudication Logic ---
    # Case 1: Both candidates exist (NEW LOGIC)
    if candidate_after and candidate_before:
        dist_after = get_vertical_distance(img_bbox, candidate_after.get('box'))
        dist_before = get_vertical_distance(img_bbox, candidate_before.get('box'))
        
        if dist_after < dist_before:
            chosen_block = candidate_after
            match_method = f"adjacent_after_closest (dist: {dist_after}px)"
            base_confidence = 0.95
        else:
            chosen_block = candidate_before
            match_method = f"adjacent_before_closest (dist: {dist_before}px)"
            base_confidence = 0.90
    
    # Case 2: Only 'after' exists
    elif candidate_after:
        chosen_block = candidate_after
        match_method = "adjacent_after"
        base_confidence = 0.95

    # Case 3: Only 'before' exists
    elif candidate_before:
        chosen_block = candidate_before
        match_method = "adjacent_before"
        base_confidence = 0.90
        
    # Case 4: No adjacent candidates, fallback to proximity
    else:
        captions_on_page = [
            b for b in blocks_by_page[page_num] 
            if b['type'] in caption_types and b['element_id'] != image_block['element_id']
        ]
        
        best_cap = None
        min_dist = float('inf')

        for cap in captions_on_page:
            dist = get_vertical_distance(img_bbox, cap.get('box'))
            if dist < min_dist:
                min_dist = dist
                best_cap = cap
                
        if min_dist < 100: # 100px threshold
            chosen_block = best_cap
            match_method = f"proximity_check (dist: {min_dist}px)"
            base_confidence = 0.50

    # --- Format output for chosen block (if any) ---
    if chosen_block:
        text = chosen_block['content']
        confidence = base_confidence
        # Boost confidence if it starts with "Figure"
        if re.search(r'^(Figure|Fig\.|图)', text.strip(), re.I):
            confidence = min(1.0, base_confidence + 0.05) # Boost
        
        return {
            "text": text, 
            "confidence": confidence, 
            "block_id": chosen_block['element_id'],
            "match_method": match_method
        }

    # No match found
    return {"text": None, "confidence": 0.0, "block_id": None, "match_method": "N/A"}


def find_table_for_caption(
    caption_text: str, 
    all_tables: List[Dict], 
    all_paragraphs: List[Dict]
) -> Dict:
    """
    Phase 3.2: Find related table using caption text.
    """
    default_info = {"table_block": None, "confidence": 0.0, "match_strategy": "N/A"}
    if not caption_text:
        return default_info

    # Strategy 1: Cross-Reference
    fig_match = re.search(r'(Figure|Fig\.|图)\s*([\d\.]+)', caption_text, re.I)
    if fig_match:
        fig_ref_text = fig_match.group(0) # e.g., "Figure 5.1"
        
        for para in all_paragraphs:
            para_text = para['content']
            tab_match = re.search(r'(Table|表)\s*([\d\.]+)', para_text, re.I)
            
            # Check if paragraph mentions BOTH the figure and a table
            if tab_match and (fig_ref_text in para_text or fig_match.group(2) in para_text):
                table_ref_text = tab_match.group(0) # e.g., "Table 3"
                table_num = tab_match.group(2)
                
                # Now find this table in all_tables
                for table in all_tables:
                    table_content_html = table['content']
                    # Check first 200 chars (likely title) for "Table 3"
                    if re.search(r'(Table|表)\s*' + re.escape(table_num), table_content_html[:200], re.I):
                        return {
                            "table_block": table, # Return the full block for now
                            "confidence": 0.90, 
                            "match_strategy": f"Cross-Reference (via {para['element_id']})"
                        }

    # Strategy 2: Keyword Match
    stop_words = {'figure', 'fig', 'table', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'with', 'on'}
    caption_keywords = set(re.findall(r'\b\w{4,}\b', caption_text.lower())) - stop_words
    
    if not caption_keywords:
        return default_info
        
    best_table = None
    best_score = 0
    
    for table in all_tables:
        table_content_html = table['content'].lower()
        score = 0
        for kw in caption_keywords:
            if kw in table_content_html:
                score += 1
        
        if score > best_score:
            best_score = score
            best_table = table
            
    # Require at least 2 strong keyword matches
    if best_score >= 2:
        return {
            "table_block": best_table,
            "confidence": 0.65,
            "match_strategy": f"Keyword Match (score: {best_score})"
        }

    return default_info


def crop_and_save_images(
    final_archives: List[Dict],
    images_map: Dict[int, Image.Image],
    output_dir: str,
    pdf_basename: str
) -> List[Dict]:
    """
    Phase 4: Crop images from PDF based on final archives.
    This function modifies `final_archives` in-place.
    """
    print(f"\nPhase 3: Cropping images...")
    crops_dir = os.path.join(output_dir, "cropped_images")
    os.makedirs(crops_dir, exist_ok=True)
    
    for archive in final_archives:
        try:
            page_num = archive['page_number']
            # Use image_bbox (standardized to pixel_bbox)
            bbox = archive.get('image_bbox') 
            
            if not bbox or len(bbox) != 4:
                print(f"  ⚠ Skipping {archive['image_id']}: bbox format error", file=sys.stderr)
                continue
                
            image = images_map.get(page_num)
            if not image:
                print(f"  ⚠ Skipping {archive['image_id']}: No image found for page {page_num}", file=sys.stderr)
                continue

            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            img_h, img_w = img_array.shape[:2]
            
            x1, y1, x2, y2 = [int(c) for c in bbox]
            
            # Add margin
            margin = 5
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img_w, x2 + margin)
            y2 = min(img_h, y2 + margin)
            
            if x1 >= x2 or y1 >= y2:
                print(f"  ⚠ Skipping {archive['image_id']}: invalid coordinates", file=sys.stderr)
                continue
                
            cropped = img_array[y1:y2, x1:x2]
            if cropped.size < 100: # Filter tiny crops
                 print(f"  ⚠ Skipping {archive['image_id']}: crop too small")
                 continue

            # Use new sequential image_id for a stable filename
            filename = f"{pdf_basename}_{archive['image_id']}.png"
            filepath = os.path.join(crops_dir, filename)
            
            cv2.imwrite(filepath, cropped)
            
            # Update the archive with the path
            archive['cropped_image_path'] = filepath
            
        except Exception as e:
            print(f"  ✗ Crop failed for {archive['image_id']}: {e}", file=sys.stderr)
            
    print(f"✓ Cropping complete. Images saved to {crops_dir}")
    return final_archives


def main():
    parser = argparse.ArgumentParser(
        description="Dolphin Result Analyzer: Processes raw JSON to extract matches and crops."
    )
    parser.add_argument(
        '--input_json', 
        type=str, 
        default='data/outputs/runpod_output/EH_01_raw_analysis.json',
        help='Path to the _raw_analysis.json file from the batch client.'
    )
    parser.add_argument(
        '--pdf_path', 
        type=str, 
        default='data/test_materials/EH_01.pdf',
        help='Path to the original PDF file (for cropping).'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/intermediate_results/EH',
        help='Output directory. (Default: a subdirectory named "processed_archives" '
             'next to the input JSON)'
    )
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='DPI for rendering PDF pages for cropping. (Default: 300)'
    )
    
    args = parser.parse_args()

    # --- 1. Setup Paths ---
    if not os.path.exists(args.input_json):
        print(f"Error: Input JSON not found '{args.input_json}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF not found '{args.pdf_path}'", file=sys.stderr)
        sys.exit(1)
        
    pdf_basename = os.path.splitext(os.path.basename(args.pdf_path))[0]
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.input_json), DEFAULT_OUTPUT_SUBDIR)
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("Dolphin Result Analyzer (Advanced Matcher V2)")
    print("="*70)
    print(f"Input JSON: {args.input_json}")
    print(f"Input PDF:  {args.pdf_path}")
    print(f"Output Dir: {output_dir}")
    print("="*70)

    # --- 2. Load and Preprocess Data ---
    print("\n[Step 1/4] Loading and preprocessing data...")
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load or parse JSON: {e}", file=sys.stderr)
        sys.exit(1)

    all_blocks_sorted, blocks_by_page, images, tables, paragraphs = preprocess_raw_data(raw_data)

    # --- 3. Core Matching Logic ---
    print("\n[Step 2/4] Assigning sequential IDs and executing matching...")
    final_archives = []
    
    # Assign sequential IDs to images and tables
    page_image_count = defaultdict(int)
    for img_block in images:
        page = img_block['page_number']
        page_image_count[page] += 1
        img_block['sequential_image_id'] = f"page_{page}_image_{page_image_count[page]}"
        
    page_table_count = defaultdict(int)
    for table_block in tables:
        page = table_block['page_number']
        page_table_count[page] += 1
        table_block['sequential_table_id'] = f"page_{page}_table_{page_table_count[page]}"

    # Create a quick lookup for global index
    element_to_global_index = {
        (block['page_number'], block['element_id']): i 
        for i, block in enumerate(all_blocks_sorted)
    }

    for image_block in images:
        global_index = element_to_global_index.get(
            (image_block['page_number'], image_block['element_id'])
        )
        if global_index is None:
            continue # Should not happen

        # Create the archive using the new sequential ID
        archive = {
            "image_id": image_block['sequential_image_id'],
            "image_bbox": image_block['box'],
            "page_number": image_block['page_number'],
            "cropped_image_path": None, # To be filled
            "caption_info": {"text": None, "confidence": 0.0, "block_id": None, "match_method": "N/A"},
            "table_match_info": {"table_info": None, "confidence": 0.0, "match_strategy": "N/A"}
        }

        # 3.1 Find Caption (New logic)
        archive['caption_info'] = find_caption_for_image(
            image_block, 
            all_blocks_sorted, 
            global_index,
            blocks_by_page
        )
        
        # 3.2 Find Table
        if archive['caption_info']['confidence'] > 0.7:
            # Pass tables (which now have sequential IDs)
            archive['table_match_info'] = find_table_for_caption(
                archive['caption_info']['text'],
                tables, 
                paragraphs
            )
            
        final_archives.append(archive)
    
    print(f"✓ Matching complete. Processed {len(final_archives)} images.")

    # --- 4. Cropping ---
    print("\n[Step 3/4] Loading PDF for image cropping...")
    images_map = get_images_for_cropping(args.pdf_path, dpi=args.dpi)
    
    final_archives = crop_and_save_images(
        final_archives, 
        images_map, 
        output_dir, 
        pdf_basename
    )

    # --- 5. Save Final Output ---
    print("\n[Step 4/4] Saving final archives file...")
    final_output_path = os.path.join(output_dir, f"{pdf_basename}_final_archives.json")
    
    try:
        # Create a deep copy for cleaning to avoid modifying list while iterating
        cleaned_archives = json.loads(json.dumps(final_archives))
        
        for archive in cleaned_archives:
            table_match = archive.get('table_match_info', {})
            
            # New logic: Clean up the table_match_info
            if table_match and table_match.get('table_block'):
                tb = table_match['table_block']
                
                # Create the new clean info block
                archive['table_match_info']['table_info'] = {
                    "table_id": tb.get('sequential_table_id', tb['element_id']),
                    "page_number": tb['page_number'],
                    "content_snippet": tb['content'][:250] + "..."
                }
                del archive['table_match_info']['table_block'] # Delete the large block
            else:
                 archive['table_match_info']['table_info'] = None
                 if 'table_block' in table_match:
                     del archive['table_match_info']['table_block']
        
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_archives, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓✓✓ All tasks complete! ✓✓✓")
        print(f"Final output saved to: {final_output_path}")

    except Exception as e:
        print(f"Error saving final JSON: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()