import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

def test_fixed_analysis():
    print("Testing Fixed OpenAI Analysis")
    print("=" * 35)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    sample_path = "samples/page_2.png"
    if not os.path.exists(sample_path):
        print(f"❌ Sample not found: {sample_path}")
        return False
    
    try:
        from src.core.region_detector import RegionDetector
        
        print("✅ Imports successful")
        
        detector = RegionDetector(api_key)
        print("✅ Detector initialized")
        
        print("🔍 Analyzing sample (this may take 15-30 seconds)...")
        result = detector.detect_regions(sample_path)
        
        print(f"✅ Analysis successful!")
        print(f"📊 Detected {len(result.regions)} regions")
        print(f"📄 Layout: {result.page_layout}")
        print(f"📐 Image size: {result.image_dimensions.width}x{result.image_dimensions.height}")
        
        # Show region types
        region_types = {}
        for region in result.regions:
            region_type = region.region_type.value
            region_types[region_type] = region_types.get(region_type, 0) + 1
        
        print("\n📋 Region breakdown:")
        for region_type, count in sorted(region_types.items()):
            print(f"  • {region_type.upper()}: {count}")
        
        print("\n🎉 OpenAI structured output working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_analysis()