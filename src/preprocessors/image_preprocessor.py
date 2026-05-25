"""
Generic Image Preprocessor (Context-Aware Version)

Features:
1. Parses flat-structure raw_analysis.json.
2. **NEW**: Collects configurable context window around each image.
3. **REMOVED**: No automatic caption matching or table cross-reference.
4. High-Resolution Cropping from original PDF.
5. Outputs archives with rich context for downstream LLM processing.
"""

import os
import sys
import json
import argparse
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


def get_images_for_cropping(pdf_path: str, dpi: int = 300) -> Dict[int, Image.Image]:
    """Loads PDF and renders pages as high-resolution images for cropping."""
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
            
            images_map[page_num + 1] = image
        
        doc.close()
        return images_map
    except Exception as e:
        print(f"Error: PDF conversion failed - {e}", file=sys.stderr)
        sys.exit(1)


def preprocess_data(raw_data: Dict) -> Tuple[List, Dict, List, List, List]:
    """Adapts to flat JSON structure and extracts elements."""
    print("Phase 1: Preprocessing data...")
    all_blocks = []
    blocks_by_page = defaultdict(list)
    images = []
    tables = []
    paragraphs = []

    type_mapping = {
        'fig': 'fig', 'figure': 'fig',
        'tab': 'tab', 'table': 'tab',
        'cap': 'cap', 'caption': 'cap',
        'header': 'header', 'sec_1': 'header',
        'para': 'para', 'text': 'para', 'list': 'list',
        'foot': 'foot', 'anno': 'anno'
    }

    results_per_page = raw_data.get('results_per_page', [])
    
    for page_data in results_per_page:
        page_num = page_data.get('page_number', 1)
        elements = page_data.get('elements', [])

        for idx, elem in enumerate(elements):
            block = {}
            block['element_id'] = elem.get('element_id', f"p{page_num}_e{idx}")
            block['page_number'] = page_num
            block['reading_order'] = elem.get('reading_order', idx)
            block['box'] = elem.get('bbox') or elem.get('pixel_bbox')
            block['content'] = elem.get('text', '')
            
            raw_label = elem.get('label', elem.get('type', 'text')).lower()
            block['type'] = type_mapping.get(raw_label, 'text')
            
            if block['content'].startswith('![Figure]'):
                block['type'] = 'fig'

            all_blocks.append(block)
            blocks_by_page[page_num].append(block)
            
            if block['type'] == 'fig':
                block['original_figure_path'] = elem.get('figure_path')
                images.append(block)
            elif block['type'] == 'tab':
                tables.append(block)
            elif block['type'] in ['para', 'header', 'list', 'cap', 'anno']:
                paragraphs.append(block)

    all_blocks_sorted = sorted(all_blocks, key=lambda b: (b.get('page_number', 0), b.get('reading_order', 0)))
    
    print(f"✓ Found: {len(all_blocks_sorted)} elements, {len(images)} images, {len(tables)} tables.")
    return all_blocks_sorted, blocks_by_page, images, tables, paragraphs


def collect_image_context(
    image_block: Dict,
    all_blocks: List[Dict],
    global_index: int,
    context_window_before: int = 3,
    context_window_after: int = 3
) -> Dict:
    """
    Collect context elements around an image.
    
    Args:
        image_block: The image element dict
        all_blocks: Sorted list of all elements
        global_index: Index of image in all_blocks
        context_window_before: Number of elements to collect before
        context_window_after: Number of elements to collect after
        
    Returns:
        Dict with context_before and context_after lists
    """
    page_num = image_block['page_number']
    context_before = []
    context_after = []
    
    # Collect elements BEFORE image
    for offset in range(1, context_window_before + 1):
        idx = global_index - offset
        if idx < 0:
            break
        
        elem = all_blocks[idx]
        
        # Stop if different page
        if elem['page_number'] != page_num:
            break
        
        # Skip other images/tables to avoid confusion
        if elem['type'] in ['fig', 'tab', 'table']:
            continue
        
        context_before.insert(0, {  # Insert at beginning to maintain order
            'element_id': elem['element_id'],
            'element_type': elem['type'],
            'text': elem['content'],
            'reading_order': elem['reading_order'],
            'position': 'before'
        })
    
    # Collect elements AFTER image
    for offset in range(1, context_window_after + 1):
        idx = global_index + offset
        if idx >= len(all_blocks):
            break
        
        elem = all_blocks[idx]
        
        # Stop if different page
        if elem['page_number'] != page_num:
            break
        
        # Skip other images/tables
        if elem['type'] in ['fig', 'tab', 'table']:
            continue
        
        context_after.append({
            'element_id': elem['element_id'],
            'element_type': elem['type'],
            'text': elem['content'],
            'reading_order': elem['reading_order'],
            'position': 'after'
        })
    
    return {
        'context_before': context_before,
        'context_after': context_after
    }


def crop_images(
    final_archives: List[Dict], 
    images_map: Dict[int, Image.Image], 
    output_dir: str, 
    basename: str
):
    """Crops and saves images."""
    crops_dir = os.path.join(output_dir, "cropped_images")
    
    # Clear existing cropped images folder if it exists
    if os.path.exists(crops_dir):
        import shutil
        print(f"Clearing existing cropped images folder: {crops_dir}")
        shutil.rmtree(crops_dir)
    
    os.makedirs(crops_dir, exist_ok=True)
    
    for archive in final_archives:
        try:
            page = archive['page_number']
            bbox = archive.get('image_bbox')
            if not bbox or not images_map.get(page):
                continue
            
            img_arr = cv2.cvtColor(np.array(images_map[page]), cv2.COLOR_RGB2BGR)
            h, w = img_arr.shape[:2]
            x1, y1, x2, y2 = [int(c) for c in bbox]
            
            margin = 5
            x1, y1 = max(0, x1-margin), max(0, y1-margin)
            x2, y2 = min(w, x2+margin), min(h, y2+margin)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            cropped = img_arr[y1:y2, x1:x2]
            filename = f"{basename}_{archive['image_id']}.png"
            path = os.path.join(crops_dir, filename)
            cv2.imwrite(path, cropped)
            archive['cropped_image_path'] = path
            
        except Exception as e:
            print(f"Crop failed for {archive['image_id']}: {e}")


def process_images_with_context(
    raw_data: Dict,
    pdf_path: str,
    output_dir: str,
    basename: str,
    context_window_before: int = 3,
    context_window_after: int = 3
) -> List[Dict]:
    """
    Main processing function with configurable context windows.
    
    Args:
        raw_data: Parsed raw_analysis.json
        pdf_path: Path to PDF file
        output_dir: Output directory
        basename: Base filename for outputs
        context_window_before: Context window before image
        context_window_after: Context window after image
        
    Returns:
        List of processed image archives
    """
    print(f"\nProcessing images with context window (before={context_window_before}, after={context_window_after})...")
    
    # 1. Preprocess data
    all_blocks, _, images, _, _ = preprocess_data(raw_data)
    
    # Build global index map
    elem_index_map = {
        (b['page_number'], b['element_id']): idx 
        for idx, b in enumerate(all_blocks)
    }
    
    # 2. Process each image
    final_archives = []
    
    for i, img in enumerate(images):
        seq_id = f"page_{img['page_number']}_img_{i+1:03d}"
        
        archive = {
            "image_id": seq_id,
            "page_number": img['page_number'],
            "image_bbox": img['box'],
            "reading_order": img['reading_order'],
            "original_figure_path": img.get('original_figure_path'),
            "context_before": [],
            "context_after": []
        }
        
        # Collect context
        global_idx = elem_index_map.get((img['page_number'], img['element_id']))
        if global_idx is not None:
            context = collect_image_context(
                img, all_blocks, global_idx,
                context_window_before, context_window_after
            )
            archive['context_before'] = context['context_before']
            archive['context_after'] = context['context_after']
        
        final_archives.append(archive)
        print(f"  Processed {seq_id}: context {len(archive['context_before'])} before, {len(archive['context_after'])} after")
    
    # 3. Crop images
    if os.path.exists(pdf_path):
        print("\nCropping images from PDF...")
        images_map = get_images_for_cropping(pdf_path)
        crop_images(final_archives, images_map, output_dir, basename)
    else:
        print("PDF not found, skipping cropping phase.")
    
    return final_archives


def main():
    parser = argparse.ArgumentParser(
        description="Image Preprocessor with Configurable Context Windows"
    )
    parser.add_argument(
        '--input_json',
        required=True,
        help='Path to raw_analysis.json'
    )
    parser.add_argument(
        '--pdf_path',
        required=True,
        help='Path to PDF file'
    )
    parser.add_argument(
        '--output_dir',
        default='output_images',
        help='Output directory'
    )
    parser.add_argument(
        '--context_before',
        type=int,
        default=3,
        help='Number of context elements before image (default: 3)'
    )
    parser.add_argument(
        '--context_after',
        type=int,
        default=3,
        help='Number of context elements after image (default: 3)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF rendering (default: 300)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"Error: {args.input_json} not found.")
        return
    
    # Load data
    with open(args.input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    basename = os.path.splitext(os.path.basename(args.pdf_path))[0]
    
    # Process images
    final_archives = process_images_with_context(
        raw_data,
        args.pdf_path,
        args.output_dir,
        basename,
        context_window_before=args.context_before,
        context_window_after=args.context_after
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{basename}_images.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "extraction_config": {
                "context_window_before": args.context_before,
                "context_window_after": args.context_after,
                "dpi": args.dpi
            },
            "total_images": len(final_archives),
            "images": final_archives
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(final_archives)} images with context to {output_path}")


if __name__ == "__main__":
    main()