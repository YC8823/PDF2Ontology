import os
import sys
import logging
import tempfile
import shutil

# -- Configure Logging --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -- Path Setup --
# This script assumes it is run from the project's root directory (pdf2ontology/).
try:
    # Get the directory of this file (examples/)
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (pdf2ontology/)
    project_root = os.path.dirname(examples_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logging.info(f"Project root '{project_root}' added to system path.")

    # Import the necessary custom modules
    from src.utils.image_utils import convert_pdf_to_images
    from src.analyzers.layout_analyzer import LayoutAnalyzer

except ImportError as e:
    logging.error(f"Failed to import a required module: {e}")
    logging.error("Please ensure you run this script from the project's root directory (pdf2ontology/).")
    sys.exit(1)
except Exception as e:
    logging.error(f"An unknown error occurred during path setup: {e}")
    sys.exit(1)


def save_images_to_directory(images, directory):
    """Saves a list of PIL Image objects to a specified directory."""
    os.makedirs(directory, exist_ok=True)
    for i, img in enumerate(images):
        output_filepath = os.path.join(directory, f"page_{i+1:02d}.png")
        try:
            img.save(output_filepath, 'PNG')
        except IOError as e:
            logging.error(f"Failed to save image '{output_filepath}': {e}")
    logging.info(f"Saved {len(images)} images to temporary directory: {directory}")


def process_pdf_for_layout_analysis(pdf_path: str, output_dir: str, model_path: str):
    """
    A complete pipeline to process a PDF for layout analysis.
    1. Converts PDF to images in a temporary directory.
    2. Analyzes images for layout information.
    3. Saves annotated results to the final output directory.
    4. Cleans up the temporary images.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Path to the final directory for annotated results.
        model_path (str): Path to the YOLO model file.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"Input PDF not found: {pdf_path}")
        return

    # Create a temporary directory to store the converted images
    # 'with' statement ensures the directory is created and available
    with tempfile.TemporaryDirectory(prefix="pdf_images_") as temp_dir:
        logging.info(f"Created temporary directory: {temp_dir}")

        # --- Step 1: Convert PDF to Images ---
        images = convert_pdf_to_images(pdf_path, dpi=300)
        if not images:
            logging.error("PDF to image conversion failed. Aborting.")
            return
        
        # Save the converted images to the temporary directory
        save_images_to_directory(images, temp_dir)

        # --- Step 2: Analyze Images for Layout ---
        logging.info("Initializing layout analyzer...")
        analyzer = LayoutAnalyzer(model_path=model_path)
        
        # Run the batch analysis using the temporary images as input
        # and the final directory as output.
        analyzer.analyze_batch(input_dir=temp_dir, output_dir=output_dir)

    # The temporary directory and its contents are automatically removed
    # upon exiting the 'with' block.
    logging.info(f"Cleanup complete. Temporary directory was removed.")
    logging.info(f"Pipeline finished. Final results are in: {output_dir}")


if __name__ == "__main__":
    # --- Configuration ---
    # This script should be run from the project's root directory.
    
    # 1. Define the input PDF file.
    INPUT_PDF_FILE = os.path.join(project_root, 'data', 'inputs', 't58700en.pdf')

    # 2. Define the final output directory.
    # The name is derived from the PDF filename.
    pdf_basename = os.path.splitext(os.path.basename(INPUT_PDF_FILE))[0]
    OUTPUT_DIR = os.path.join(project_root, 'data', 'outputs', f"{pdf_basename}_with_layout_analysis")
    
    # 3. Define the path to the YOLO model.
    MODEL_PATH = os.path.join(project_root, 'src', 'analyzers', 'yolov10s_best.pt')

    # --- Execution ---
    process_pdf_for_layout_analysis(
        pdf_path=INPUT_PDF_FILE,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH
    )
