
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict

# Setup project path
sys.path.insert(0, str(Path.cwd()))

def safe_access(obj, attr, default=None):
    """Safely access attribute from dict or object"""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    elif isinstance(obj, dict) and attr in obj:
        return obj[attr]
    return default

def test_region_detector():
    """Simple test showing GPT output and timing"""
    
    print("Region Detector Test")
    print("=" * 30)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    # Check sample file
    sample_path = "samples/page_1_resized_processed.png"
    if not os.path.exists(sample_path):
        print(f"❌ Sample not found: {sample_path}")
        return False
    
    print(f"📄 Sample: {sample_path}")
    print(f"🔑 API Key: ...{api_key[-4:]}")
    print()
    
    try:
        # Import detector
        from src.core.region_detector import RegionDetector
        
        # Initialize detector
        print("🔧 Initializing detector...")
        detector = RegionDetector(api_key)
        
        # Run detection with timing
        print("🧠 Running GPT analysis...")
        start_time = time.time()
        
        result = detector.detect_regions(sample_path)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Analysis completed in {total_time:.2f} seconds")
        print()
        
        # Show GPT output
        print("🤖 GPT OUTPUT:")
        print("-" * 40)
        
        print(f"Document ID: {result.document_id}")
        print(f"Total regions: {len(result.regions)}")

        # Handle image_dimensions safely
        img_dims = safe_access(result, 'image_dimensions', {})
        if img_dims:
            width = safe_access(img_dims, 'width', 'unknown')
            height = safe_access(img_dims, 'height', 'unknown')
            print(f"Image size: {width}x{height}")
        print()

        # print(f"Image size: {result.image_dimensions.width}x{result.image_dimensions.height}")
        # print()
        
        # Page layout
        page_layout = safe_access(result, 'page_layout', 'No layout description')
        print("Page Layout Description:")
        print(f"  {page_layout}")
        print()
        
        # Processing metadata
        proc_meta = safe_access(result, 'processing_metadata', {})
        if proc_meta:
            doc_type = safe_access(proc_meta, 'document_type')
            if doc_type:
                print(f"Document Type: {doc_type}")
            
            complexity = safe_access(proc_meta, 'complexity_score')
            if complexity:
                print(f"Complexity Score: {complexity}")
            print()
        
        # Detected regions
        regions = safe_access(result, 'regions', [])
        if regions:
            print("Detected Regions:")
            for i, region in enumerate(regions, 1):
                # Region ID and type
                region_id = safe_access(region, 'region_id', f'region_{i}')
                
                # Handle region type (enum or string)
                region_type = safe_access(region, 'region_type', 'unknown')
                if hasattr(region_type, 'value'):
                    region_type = region_type.value
                region_type_str = str(region_type).upper()
                
                print(f"  {i}. {region_id} ({region_type_str})")
                
                # Bounding box
                bbox = safe_access(region, 'bbox', {})
                if bbox:
                    x = safe_access(bbox, 'x', 0)
                    y = safe_access(bbox, 'y', 0)
                    width = safe_access(bbox, 'width', 0)
                    height = safe_access(bbox, 'height', 0)
                    print(f"     Position: ({x}, {y}) Size: {width}x{height}")
                
                # Confidence
                confidence = safe_access(region, 'confidence', 0.0)
                print(f"     Confidence: {confidence:.3f}")
                
                # Content description
                content_desc = safe_access(region, 'content_description', 'No description')
                print(f"     Content: {content_desc}")
                
                # Reading order
                reading_order = safe_access(region, 'reading_order')
                if reading_order:
                    print(f"     Reading Order: {reading_order}")
                print()
        
        # Show summary
        print("📊 SUMMARY:")
        print(f"   ⏱️  Total time: {total_time:.2f} seconds")
        print(f"   📍 Regions found: {len(result.regions)}")
        
        # Region type breakdown
        region_types = {}
        for region in result.regions:
            region_type = region.region_type.value
            region_types[region_type] = region_types.get(region_type, 0) + 1
        
        print("   📋 Region types:")
        for region_type, count in sorted(region_types.items()):
            print(f"      • {region_type}: {count}")
        
        # Average confidence
        avg_confidence = sum(r.confidence for r in result.regions) / len(result.regions)
        print(f"   🎯 Average confidence: {avg_confidence:.3f}")
        
        # Save raw output to file
        output_file = "gpt_raw_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, default=str, ensure_ascii=False)
        
        print(f"   💾 Raw output saved: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_region_detector()