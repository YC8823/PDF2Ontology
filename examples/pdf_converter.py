import os
import sys
import logging
from PIL import Image

# -- Configure Logging --
# Set up basic logging to see informational messages in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -- Path Setup --
# This script assumes it is run from the project's root directory (pdf2ontology/).
# We add the project root to the system path to ensure module imports work correctly.
try:
    # Get the directory of this file (examples/)
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (pdf2ontology/)
    project_root = os.path.dirname(examples_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logging.info(f"Project root '{project_root}' added to system path.")

    # Now we can safely import from src.utils
    from src.utils.image_utils import convert_pdf_to_images

except ImportError as e:
    logging.error(f"Failed to import module: {e}")
    logging.error("Please ensure you run this script from the project's root directory (pdf2ontology/), e.g., python examples/pdf_converter.py")
    sys.exit(1)
except Exception as e:
    logging.error(f"An unknown error occurred during path setup: {e}")
    sys.exit(1)


def convert_pdf_and_save(pdf_path: str, output_dir: str, dpi: int = 300):
    """
    Loads a PDF file, converts its pages to images, and saves them to a specified directory.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_dir (str): The directory path to save the output images.
        dpi (int): The resolution (dots per inch) to use for the conversion.
    """
    # 1. Check if the input PDF file exists
    if not os.path.exists(pdf_path):
        logging.error(f"Input file not found: {pdf_path}")
        return

    # 2. Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory created or already exists: {output_dir}")
    except OSError as e:
        logging.error(f"Failed to create directory '{output_dir}': {e}")
        return

    # 3. Call the utility function to convert the PDF to a list of PIL Image objects
    logging.info(f"Starting conversion of PDF file: {pdf_path}...")
    images = convert_pdf_to_images(pdf_path, dpi=dpi)

    if not images:
        logging.warning("PDF conversion did not produce any images. Please check the PDF file or logs from 'image_utils'.")
        return

    # 4. Iterate through all images and save them in PNG format
    for i, img in enumerate(images):
        # Create a filename for each image (e.g., page_01.png, page_02.png)
        output_filename = f"page_{i+1:02d}.png"
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            # Save the image
            img.save(output_filepath, 'PNG')
            logging.info(f"Successfully saved image: {output_filepath}")
        except IOError as e:
            logging.error(f"Failed to save image '{output_filepath}': {e}")

    logging.info("All pages have been successfully converted and saved!")


if __name__ == "__main__":
    # -- Define File Paths --
    # Use os.path.join to ensure paths work correctly on different operating systems
    
    # Path to the input PDF file
    input_pdf_file = os.path.join(project_root, 'data', 'inputs', 't58700en.pdf')

    # Path for the output directory
    # We get the base name from the input filename (sample01) and add a suffix
    pdf_basename = os.path.splitext(os.path.basename(input_pdf_file))[0]
    output_directory = os.path.join(project_root, 'data', 'outputs', f"{pdf_basename}_converted")

    # -- Run the Conversion --
    convert_pdf_and_save(input_pdf_file, output_directory)
