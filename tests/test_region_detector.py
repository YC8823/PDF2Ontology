"""
Simple test script for region detector
Run with: python test_region_detector.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is in Python path to import src module
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_region_detector():
    """Test the region detector with a sample image"""
    
    from dotenv import load_dotenv
    load_dotenv()

    # You'll need to set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Import after path setup
        from src.core.region_detector import RegionDetector
        
        # Initialize detector
        print("🔧 Initializing RegionDetector...")
        detector = RegionDetector(api_key)

        image_path = "sample/page_3.png"  
        
        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            print("💡 Please update image_path in the script to point to a real image")
            return
        
        print(f"📄 Analyzing image: {image_path}"   )
        
        # Analyze document layout
        layout = detector.analyze_document(image_path, page_number=1)
        
        print("✅ Analysis completed!")
        print(f"📊 Results:")
        print(f"   - Total regions detected: {len(layout.regions)}")
        print(f"   - Page number: {layout.page_number}")
        
        # Show region details
        if layout.regions:
            print(f"\n📋 Region Details:")
            for i, region in enumerate(layout.regions[:5]):  # Show first 5
                print(f"   {i+1}. {region.id}")
                print(f"      Type: {region.type.value}")
                print(f"      Confidence: {region.confidence:.2f}")
                print(f"      Description: {region.content_description[:50]}...")
                print(f"      Bbox: ({region.bbox.x:.3f}, {region.bbox.y:.3f}, "
                      f"{region.bbox.width:.3f}, {region.bbox.height:.3f})")
                print()
            
            if len(layout.regions) > 5:
                print(f"   ... and {len(layout.regions) - 5} more regions")
        
        print("🎉 Test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the correct directory")
        print("💡 Check that all required packages are installed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_document_analyzer():
    """Test the full document analyzer"""

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Import document analyzer
        from src.core.document_analyzer import DocumentAnalyzer
        
        print("🔧 Initializing DocumentAnalyzer...")
        analyzer = DocumentAnalyzer(api_key)
        
        # Test with sample file
        input_path = "data/inputs/sample_Datenblatt.pdf"  # Update this path
        output_dir = "data/outputs/test_results"
        
        if not os.path.exists(input_path):
            print(f"❌ Test file not found: {input_path}")
            print("💡 Please update input_path to point to a real PDF or image")
            return
        
        print(f"📄 Analyzing document: {input_path}")
        
        # Run analysis
        results = analyzer.analyze_document(
            input_path=input_path,
            output_dir=output_dir,
            extract_tables=True,
            create_visualizations=True
        )
        
        print("✅ Analysis completed!")
        print(f"📊 Results Summary:")
        print(f"   - Total pages: {results['document_info']['total_pages']}")
        print(f"   - Processing time: {results['document_info']['processing_time']:.2f}s")
        print(f"   - Total regions: {results['analysis_summary']['total_regions_detected']}")
        print(f"   - Table regions: {results['analysis_summary']['total_table_regions']}")
        print(f"   - Successful tables: {results['analysis_summary']['successful_table_analyses']}")
        
        print(f"\n📁 Output files saved to: {output_dir}")
        if 'output_files' in results:
            for file_type, file_path in results['output_files'].items():
                print(f"   - {file_type}: {file_path}")
        
        print("🎉 Full test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the correct directory")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Testing PDF2Ontology Components")
    print("=" * 50)
    
    # Test 1: Region Detector only
    print("\n🔍 Test 1: RegionDetector")
    print("-" * 30)
    test_region_detector()
    
    # Test 2: Full Document Analyzer
    print("\n📄 Test 2: DocumentAnalyzer")  
    print("-" * 30)
    test_document_analyzer()
    
    print("\n✨ All tests completed!")