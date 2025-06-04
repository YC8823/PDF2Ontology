import os
import sys
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

def setup_project_path():
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    return current_dir

def test_imports_fixed():
    print("Testing Imports (Fixed)")
    print("=" * 30)
    
    project_root = setup_project_path()
    print(f"Project root: {project_root}")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    try:
        # Test core imports
        from src.core.models import RegionType, BoundingBox, DocumentRegion
        print("✅ Core models imported")
        
        from src.core.region_detector import RegionDetector
        print("✅ Structured detector imported")
        
        from src.core.document_analyzer import DocumentAnalyzer
        print("✅ Enhanced analyzer imported")
        
        # Test initialization
        analyzer = DocumentAnalyzer(api_key)
        print("✅ Analyzer initialized")
        
        print("\n🎉 All imports working!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports_fixed()