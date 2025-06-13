"""
Simple example using your existing sample image
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.document_analyzer import DocumentAnalyzer

def analyze_sample_image():
    """Analyze the existing sample image using structured output"""
    
    print("Document Analysis Example - Real Sample")
    print("=" * 50)
    
    # Configuration
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable required")
        print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # File paths
    sample_image = "samples/page_3.png"
    output_dir = "results/sample_analysis"
    
    # Check if sample exists
    if not os.path.exists(sample_image):
        print(f"❌ Error: Sample image not found at {sample_image}")
        print("   Please ensure the file exists in the samples/ directory")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📄 Input: {sample_image}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Using: GPT-4o with structured output")
    print()
    
    try:
        # Initialize analyzer
        print("🚀 Initializing analyzer...")
        analyzer = DocumentAnalyzer(openai_api_key=api_key)
        
        # Run analysis with structured output
        print("🔍 Analyzing document regions...")
        result = analyzer.analyze_document(
            input_path=sample_image,
            output_dir=output_dir,
            enhance_image=True,
            annotation_style="detailed",
            extract_tables=True
        )
     
        #     extract_content=True
        # )
        
        print("✅ Analysis completed successfully!")
        print()
        
        # Display main results
        analysis = result["analysis_result"]
        regions = analysis["regions"]
        
        print("📊 ANALYSIS SUMMARY")
        print("-" * 30)
        print(f"Total regions detected: {len(regions)}")
        print(f"Page layout: {analysis['page_layout']}")
        print(f"Image dimensions: {analysis['image_dimensions']['width']}x{analysis['image_dimensions']['height']}")
        
        # Show processing metadata if available
        metadata = analysis.get('processing_metadata', {})
        if metadata:
            print(f"Document type: {metadata.get('document_type', 'unknown')}")
        
        # Region breakdown
        region_types = {}
        for region in regions:
            region_type = region['region_type']
            region_types[region_type] = region_types.get(region_type, 0) + 1
        
        print("\n📋 DETECTED REGIONS")
        print("-" * 30)
        for region_type, count in sorted(region_types.items()):
            print(f"• {region_type.upper()}: {count} region(s)")
        
        # Show first few regions with details
        print("\n🔍 REGION DETAILS (First 5)")
        print("-" * 30)
        for i, region in enumerate(regions[:5]):
            bbox = region['bbox']
            print(f"{i+1}. {region['region_id']} ({region['region_type'].upper()})")
            print(f"   Position: ({bbox['x']}, {bbox['y']}) Size: {bbox['width']}x{bbox['height']}")
            print(f"   Confidence: {region['confidence']:.3f}")
            print(f"   Content: {region['content_description'][:60]}...")
            if region.get('reading_order'):
                print(f"   Reading order: {region['reading_order']}")
            print()
        
        # Table analysis results
        if result["table_analysis"]:
            print("📋 TABLE ANALYSIS")
            print("-" * 30)
            tables = result["table_analysis"]
            print(f"Tables found: {len(tables)}")
            for table_id, table_data in tables.items():
                print(f"• {table_id}: {table_data['rows']} rows × {table_data['columns']} columns")
                if table_data.get('table_caption'):
                    print(f"  Caption: {table_data['table_caption']}")
                print(f"  Confidence: {table_data['confidence']:.3f}")
        
        # Content extraction results
        # if result["content_extraction"]:
        #     print("\n📝 CONTENT EXTRACTION")
        #     print("-" * 30)
        #     content = result["content_extraction"]
        #     print(f"Total text length: {content.get('total_text_length', 0)} characters")
        #     print(f"Total words: {content.get('total_word_count', 0)}")
        #     print(f"Primary language: {content.get('primary_language', 'unknown')}")
            
        #     text_regions = content.get('text_regions', [])
        #     if text_regions:
        #         print(f"Text regions processed: {len(text_regions)}")
        #         for text_region in text_regions[:3]:  # Show first 3
        #             print(f"• {text_region['region_id']}: {text_region['word_count']} words")
        #             print(f"  Quality: {text_region['text_quality']:.3f}")
        #             if len(text_regions) > 3:
        #                 print(f"  ... and {len(text_regions) - 3} more")
        #                 break
        
        # Output files
        print(f"\n📄 OUTPUT FILES")
        print("-" * 30)
        output_files = result["output_files"]
        for file_type, file_path in output_files.items():
            file_name = os.path.basename(file_path)
            print(f"• {file_type}: {file_name}")
        
        # Processing time
        processing_time = result["metadata"]["processing_time"]
        print(f"\n⏱️  Total processing time: {processing_time:.2f} seconds")
        
        print(f"\n🖼️  View annotated image: {output_files.get('annotated_image', 'N/A')}")
        print(f"📊 Full results JSON: {output_files.get('structured_results', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        print("\nFull error details:")
        print(traceback.format_exc())
        return False

def quick_analysis():
    """Quick analysis with minimal output"""
    
    print("Quick Analysis of Sample Image")
    print("=" * 40)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY required")
        return
    
    sample_image = "samples/page_2.png"
    if not os.path.exists(sample_image):
        print(f"❌ {sample_image} not found")
        return
    
    try:
        from src.core.region_detector import RegionDetector
        
        # Quick region detection only
        detector = RegionDetector(api_key)
        result = detector.detect_regions(sample_image)
        
        print(f"✅ Detected {len(result.regions)} regions")
        print(f"📄 Layout: {result.page_layout}")
        
        for region in result.regions:
            print(f"• {region.region_type.value}: {region.content_description[:50]}...")
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {str(e)}")

def export_regions_to_json():
    """Export just the region data to a clean JSON file"""
    
    print("Exporting Region Data to JSON")
    print("=" * 40)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY required")
        return
    
    sample_image = "samples/page_2.png"
    if not os.path.exists(sample_image):
        print(f"❌ {sample_image} not found")
        return
    
    try:
        from src.core.region_detector import RegionDetector
        
        detector = RegionDetector(api_key)
        result = detector.detect_regions(sample_image)
        
        # Create clean export data
        export_data = {
            "document_info": {
                "source_file": sample_image,
                "total_regions": len(result.regions),
                "image_dimensions": result.image_dimensions,
                "analysis_timestamp": result.timestamp.isoformat()
            },
            "layout_description": result.page_layout,
            "regions": []
        }
        
        # Add region data
        for region in result.regions:
            region_data = {
                "id": region.region_id,
                "type": region.region_type.value,
                "coordinates": {
                    "x": region.bbox.x,
                    "y": region.bbox.y,
                    "width": region.bbox.width,
                    "height": region.bbox.height
                },
                "confidence": round(region.confidence, 3),
                "description": region.content_description,
                "reading_order": region.reading_order
            }
            export_data["regions"].append(region_data)
        
        # Save to file
        output_file = "sample_regions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(result.regions)} regions to {output_file}")
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze sample document image")
    parser.add_argument("--mode", choices=["full", "quick", "export"], default="full",
                       help="Analysis mode: full (complete), quick (regions only), export (JSON only)")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        success = analyze_sample_image()
        if success:
            print("\n🎉 Analysis completed successfully!")
        else:
            print("\n💥 Analysis failed. Check error messages above.")
    
    elif args.mode == "quick":
        quick_analysis()
    
    elif args.mode == "export":
        export_regions_to_json()

