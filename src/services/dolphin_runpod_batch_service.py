# src/services/dolphin_runpod_batch_service.py
"""
Dolphin RunPod Batch Processing Client (Fixed for File Upload)

This client's sole responsibility is to:
1. Convert a PDF to images.
2. Send each image to the RunPod server for analysis using multipart/form-data.
3. Collect all raw JSON responses.
4. Save the combined raw JSON to a single file.
"""

import os
import sys
import json
import argparse
import requests
import time
import io
from typing import List, Dict, Tuple
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: 'PyMuPDF' library not found.")
    print("Please install: pip install PyMuPDF")
    sys.exit(1)

# API URL (assuming SSH tunnel or direct access)
# 确保端口与 SSH 隧道一致 (8080)
API_URL = "http://localhost:8080/analyze"

# Default output directory
DEFAULT_OUTPUT_DIR = "./data/outputs/runpod_output"


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Tuple[Image.Image, int]]:
    """
    Converts all pages of a PDF to a list of images.
    """
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        print(f"PDF Page Count: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Calculate zoom factor from DPI
            zoom = dpi / 72  # Default is 72 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            if pix.alpha:
                image = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
                image = image.convert("RGB")
            else:
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            images.append((image, page_num + 1))  # Page numbers start at 1
            print(f"  Converted page {page_num + 1}/{len(doc)} (Size: {image.size})")
        
        doc.close()
        print(f"✓ PDF conversion complete: {len(images)} pages.")
        return images
        
    except Exception as e:
        print(f"Error: PDF conversion failed - {e}", file=sys.stderr)
        sys.exit(1)


def send_batch_to_runpod(
    images_with_pages: List[Tuple[Image.Image, int]], 
    api_url: str,
    batch_size: int = 1
) -> List[Dict]:
    """
    Sends images to RunPod for analysis using standard File Upload.
    """
    results = []
    total_images = len(images_with_pages)
    
    print(f"\nStarting batch analysis of {total_images} pages (batch size: {batch_size})...")
    
    for i in range(0, total_images, batch_size):
        batch = images_with_pages[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_images + batch_size - 1) // batch_size
        
        print(f"\nBatch {batch_num}/{total_batches}: Processing pages {i+1} to {min(i+batch_size, total_images)}...")
        
        for j, (img, page_num) in enumerate(batch):
            start_time = time.time()
            try:
                print(f"  [{i+j+1}/{total_images}] Analyzing page {page_num}...")
                
                # === 修改开始: 将图片转换为二进制流 ===
                img_byte_arr = io.BytesIO()
                # 保存为 PNG 格式
                img.save(img_byte_arr, format='PNG')
                # 重置指针到开始位置
                img_byte_arr.seek(0)
                
                # 构造 multipart/form-data 请求
                # key 必须是 'file'，以匹配服务端 @app.post("/analyze") 中的参数名
                files = {
                    'file': (f'page_{page_num}.png', img_byte_arr, 'image/png')
                }
                
                # 发送请求 (使用 files 参数而不是 json 参数)
                response = requests.post(
                    api_url,
                    files=files,
                    timeout=300
                )
                # === 修改结束 ===

                response.raise_for_status()
                
                result = response.json()
                
                # 确保结果包含页码 (客户端维护页码信息)
                # 服务端返回的单页结果通常也是个列表或对象，这里做兼容处理
                if isinstance(result, dict):
                    if 'results_per_page' in result and isinstance(result['results_per_page'], list):
                        # 提取内部的元素列表
                        first_page_result = result['results_per_page'][0]
                        # 合并到外层结果中，或者直接使用
                        if 'elements' in first_page_result:
                            result['elements'] = first_page_result['elements']
                    
                    result['page_number'] = page_num
                
                results.append(result)
                
                duration = time.time() - start_time
                # 尝试获取元素数量用于显示
                num_elements = 0
                if 'elements' in result:
                    num_elements = len(result['elements'])
                elif 'results_per_page' in result:
                    num_elements = len(result['results_per_page'][0].get('elements', []))

                print(f"    ✓ Page {page_num}: OK ({num_elements} elements) in {duration:.2f}s")
                
            except requests.exceptions.Timeout:
                print(f"    ✗ Page {page_num}: FAILED (Timeout)", file=sys.stderr)
                results.append({"error": "timeout", "page_number": page_num})
            except Exception as e:
                # 打印详细的错误响应内容，方便调试
                error_msg = str(e)
                if 'response' in locals() and hasattr(response, 'text'):
                     print(f"      Server Response: {response.text}", file=sys.stderr)
                
                print(f"    ✗ Page {page_num}: FAILED ({error_msg})", file=sys.stderr)
                results.append({"error": error_msg, "page_number": page_num})
    
    successful = len([r for r in results if 'error' not in r])
    print(f"\n✓ Batch analysis complete: {successful}/{total_images} pages successful.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Dolphin RunPod Batch Client: Converts PDF and gets raw analysis."
    )
    parser.add_argument(
        '--pdf_path', 
        type=str, 
        required=True,
        help='Path to the PDF file to analyze.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for the raw analysis JSON. (Default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='DPI for rendering PDF pages. (Default: 300)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='Number of pages to process in a single batch. (Default: 1)'
    )
    parser.add_argument(
        '--api_url', 
        type=str, 
        default=API_URL,
        help=f'RunPod server endpoint URL. (Default: {API_URL})'
    )
    
    args = parser.parse_args()
    
    # Validation
    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found '{args.pdf_path}'", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    pdf_basename = os.path.splitext(os.path.basename(args.pdf_path))[0]
    
    print("="*70)
    print("Dolphin RunPod Batch Client (Multipart Upload)")
    print("="*70)
    print(f"PDF File: {args.pdf_path}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Render DPI: {args.dpi}")
    print(f"Batch Size: {args.batch_size}")
    print(f"API URL: {args.api_url}")
    print("="*70)
    
    # Step 1: PDF -> Images
    print("\n[Step 1/3] Converting PDF to images...")
    images_with_pages = convert_pdf_to_images(args.pdf_path, dpi=args.dpi)
    
    # Step 2: Batch Analysis
    print("\n[Step 2/3] Sending images for batch analysis...")
    analysis_results = send_batch_to_runpod(
        images_with_pages, 
        args.api_url,
        args.batch_size
    )
    
    # Step 3: Save Raw Analysis Results
    print("\n[Step 3/3] Saving raw analysis results...")
    results_path = os.path.join(args.output_dir, f"{pdf_basename}_raw_analysis.json")
    
    try:
        # 结果结构整理，使其尽量符合 SS_03_raw_analysis.json 的多页结构
        final_results = []
        for res in analysis_results:
            # 清理错误项
            if 'error' in res:
                continue
            
            # 如果服务端返回的是完整列表结构 (results_per_page)，我们取其中的元素
            if 'results_per_page' in res and isinstance(res['results_per_page'], list):
                page_data = res['results_per_page'][0]
                # 注入正确的页码
                page_data['page_number'] = res.get('page_number', 0)
                final_results.append(page_data)
            else:
                # 否则直接追加
                final_results.append(res)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'pdf_filename': os.path.basename(args.pdf_path),
                'total_pages': len(images_with_pages),
                'results_per_page': final_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ All processing complete!")
        print(f"  Raw analysis JSON saved to: {results_path}")
        print(f"\nNext step: Run the analyzer on this file:")
        print(f"python src/analyzers/dolphin_result_analyzer.py --input_json {results_path} --pdf_path {args.pdf_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()