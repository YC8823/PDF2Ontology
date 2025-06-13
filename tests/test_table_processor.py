import sys
import os
from PIL import Image
import pandas as pd
import requests

# Ensure the src directory is on the import path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    from processors.table_processor import TableProcessor
except ImportError:
    print("Error: Could not import TableProcessor.")
    print("Please make sure 'src/processors/table_processor.py' exists and the script is run from the project root.")
    sys.exit(1)

def run_test():
    """
    Runs a standalone test for the TableProcessor.
    """
    print("--- Starting TableProcessor Test ---")

    # --- 1. Initialization ---
    print("\n[Step 1/4] Initializing TableProcessor...")
    try:
        # Instantiate the processor. This will load the ML models into memory.
        table_processor = TableProcessor()
    except Exception as e:
        print(f"Failed to initialize TableProcessor: {e}")
        return

    # --- 2. Load Test Image ---
    print("\n[Step 2/4] Loading test image...")
    # An online image URL is used for easy, direct execution.
    # You can replace this with a local file path.
    image_path = "samples/page_3_processed.png"
    # image_url = "https://i.imgur.com/4z33b0y.png"
    
    try:
        # Load the image from the URL using Pillow
        image = Image.open(image_path).convert("RGB") # 直接用Image.open打开
        print(f"Successfully loaded image from local path: {image_path}")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # --- 3. Execute Processing ---
    print("\n[Step 3/4] Calling extract_structured_table()...")
    # Call the main method to extract the table
    result_df = table_processor.extract_structured_table(image)

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