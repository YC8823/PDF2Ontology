import os
import sys 
# 确保项目根目录在 Python 路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import json
from pathlib import Path

# 1) Adjust this import to match your project structure:
from src.visual_analyzer import detect_layout_regions, visualize_layout

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input-image> <output-image>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}")
        sys.exit(1)

    # (Optional) ensure your OPENAI_API_KEY is set, e.g.
    #   export OPENAI_API_KEY=sk-...
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY not set in environment; the model call will fail.")

    print("► Running layout detection on:", input_path)
    try:
        regions = detect_layout_regions(str(input_path))
    except Exception as e:
        print("Layout detection failed:", e)
        sys.exit(2)

    print("► Detected regions (JSON):")
    print(json.dumps(regions, indent=2))

    print(f"► Annotating image and saving to {output_path} …")
    visualize_layout(str(input_path), regions, str(output_path))

    if output_path.exists():
        print("✅ Annotated image written to:", output_path)
    else:
        print("❌ Failed to write annotated image.")

if __name__ == "__main__":
    main()