import sys
import os
from PIL import Image
import pandas as pd
import requests

# Ensure project root is in Python path to import src module
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# try:
#     from src.processors.table_processor import TableProcessor
# except ImportError:
#     print("Error: Could not import TableProcessor.")
#     print("Please make sure 'src/processors/table_processor.py' exists and the script is run from the project root.")
#     sys.exit(1)
from table_processors.table_transformer import TableTransformerProcessor
from src.utils.image_utils import convert_pdf_to_images

def run_test():
    """
    Runs a standalone test for the TableProcessor.
    """
    print("--- Starting TableProcessor Test ---")

    # --- 1. Initialization ---
    print("\n[Step 1/4] Initializing TableProcessor...")
    try:
        # Instantiate the processor. This will load the ML models into memory.
        table_processor = TableTransformerProcessor()
    except Exception as e:
        print(f"Failed to initialize TableProcessor: {e}")
        return

    # --- 2. Load Test Image ---
    print("\n[Step 2/4] Loading test image...")
    # An online image URL is used for easy, direct execution.
    # You can replace this with a local file path.
    # image_path = "samples/page_3_processed.png"
    pdf_path = "data/inputs/sample02.pdf"
    
    # try:
    #     # Load the image from the URL using Pillow
    #     image = Image.open(image_path).convert("RGB") # 直接用Image.open打开
    #     print(f"Successfully loaded image from local path: {image_path}")
    # except Exception as e:
    #     print(f"Failed to load image: {e}")
    #     return
    if not os.path.exists(pdf_path):
        print(f"Test PDF not found at '{pdf_path}'. Please update the path.")
        return
        
    # Use the new utility to convert PDF to a list of enhanced images
    page_images = convert_pdf_to_images(pdf_path)
    
    if not page_images:
        print("No pages were converted from the PDF. Aborting test.")
        return

    # --- 3. Execute Processing ---
    # print("\n[Step 3/4] Calling extract_structured_table()...")
    # # Call the main method to extract the table
    # result_df = table_processor.extract_structured_table(
    # page_image,
    # debug_prefix="debug_run"  # 激活调试模式并设置文件名前缀
    # )
    
     # --- 3. Process Each Page ---
    print("\n[Step 2/3] Processing each page from the PDF...")
    all_tables = []
    for i, page_image in enumerate(page_images):
        page_num = i + 1
        print(f"\n--- Processing Page {page_num} ---")
        
        # Call the main method to extract the table from the current page
        result_df = table_processor.extract_structured_table(page_image)
        
        if result_df is not None and not result_df.empty:
            print(f"✅ Table found and extracted on page {page_num}.")
            all_tables.append({"page": page_num, "table": result_df})
        else:
            print(f"ℹ️ No table found on page {page_num}.")

    # --- 4. Validation ---
    print("\n[Step 4/4] Validating the output...")
    try:
        # Test 1: Check if the output is a Pandas DataFrame
        assert isinstance(result_df, pd.DataFrame), f"Test Failed: Output is not a DataFrame, but {type(result_df)}."
        print("✅ PASSED: Output is a DataFrame.")

        # Test 2: Check if the DataFrame is not empty
        assert not result_df.empty, "Test Failed: The extracted DataFrame is empty."
        print("✅ PASSED: DataFrame is not empty.")
        
        # Test 3: Check if the DataFrame has a reasonable number of rows and columns
        assert result_df.shape[0] > 0, "Test Failed: DataFrame has no rows."
        assert result_df.shape[1] > 0, "Test Failed: DataFrame has no columns."
        print(f"✅ PASSED: DataFrame has a shape of {result_df.shape} (rows, columns).")
        
        print("\n--- All Basic Tests Passed Successfully! ---")
        print("\nFinal Extracted DataFrame:")
        print("------------------------------------------")
        print(result_df)
        print("------------------------------------------")

    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during validation: {e}")


if __name__ == '__main__':
    # Make sure you have installed all dependencies:
    # pip install transformers torch pandas pillow requests pytesseract
    # Also, ensure the Tesseract OCR engine is installed on your system.
    run_test()