import os
import sys
import logging
import json
import glob
import cv2
from ultralytics import YOLO

# -- Configure Logging --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -- Path Setup --
# This allows the script to be run from the project root (pdf2ontology/)
# and correctly import other modules if needed in the future.
try:
    # The directory containing this script (src/analyzer/)
    analyzer_dir = os.path.dirname(os.path.abspath(__file__))
    # The source directory (src/)
    src_dir = os.path.dirname(analyzer_dir)
    # The project root directory (pdf2ontology/)
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logging.info(f"Project root '{project_root}' added to system path.")
except Exception as e:
    logging.error(f"An unknown error occurred during path setup: {e}")
    sys.exit(1)

class LayoutAnalyzer:
    """
    A class to analyze document layouts in images using a YOLO model.
    It processes images in batch, saving annotated images and structured layout data.
    """
    def __init__(self, model_path: str):
        """
        Initializes the LayoutAnalyzer with a pre-trained YOLO model.

        Args:
            model_path (str): The local file path to the YOLO model (.pt file).
        """
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Loads the YOLO model from the specified path."""
        try:
            logging.info(f"Loading YOLO model from: {self.model_path}...")
            model = YOLO(self.model_path)
            logging.info("YOLO model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Failed to load YOLO model from '{self.model_path}': {e}")
            sys.exit(1)

    def analyze_batch(self, input_dir: str, output_dir: str):
        """
        Analyzes a batch of images from an input directory and saves the results.

        Args:
            input_dir (str): Directory containing the images to analyze (e.g., PNG files).
            output_dir (str): Directory to save annotated images and JSON layout data.
        """
        if not os.path.isdir(input_dir):
            logging.error(f"Input directory not found: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output will be saved to: {output_dir}")

        image_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        if not image_paths:
            logging.warning(f"No PNG images found in directory: {input_dir}")
            return

        logging.info(f"Found {len(image_paths)} images to analyze.")

        for image_path in image_paths:
            self._process_single_image(image_path, output_dir)
        
        logging.info("Batch analysis and annotation complete.")

    def _process_single_image(self, image_path: str, output_dir: str):
        """Processes a single image, annotates it, and saves the layout data."""
        try:
            logging.info(f"Processing: {os.path.basename(image_path)}")
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                logging.warning(f"Could not read image {image_path}, skipping.")
                return

            # Perform prediction
            results = self.model.predict(source=img_cv, save=False, verbose=False)
            result = results[0]
            
            layout_data = {"page": os.path.basename(image_path), "layouts": []}
            
            # Draw bounding boxes and collect data
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Store layout information
                layout_data["layouts"].append({
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2]
                })

                # Draw annotation on the image
                self._draw_annotation(img_cv, label, confidence, x1, y1, x2, y2)

            # Save the annotated image and the JSON data
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            annotated_image_path = os.path.join(output_dir, f"{base_filename}_annotated.png")
            cv2.imwrite(annotated_image_path, img_cv)

            json_path = os.path.join(output_dir, f"{base_filename}_layout.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(layout_data, f, indent=4)

        except Exception as e:
            logging.error(f"An error occurred while processing {image_path}: {e}")

    @staticmethod
    def _draw_annotation(image, label, conf, x1, y1, x2, y2):
        """Helper function to draw a single annotation box and label on an image."""
        color = (255, 0, 0)  # Blue in BGR
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        text = f'{label} {conf:.2f}'
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_w, text_h = text_size
        
        # Position the label background and text
        rect_y = y1 - text_h - 10
        text_y = y1 - 5
        if rect_y < 0:
            rect_y = y1 + 10
            text_y = y1 + text_h + 10

        cv2.rectangle(image, (x1, rect_y), (x1 + text_w, rect_y + text_h + 5), color, -1)
        cv2.putText(image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


if __name__ == "__main__":
    # --- Configuration ---
    # This script should be run from the project root directory.
    
    # Directory containing the images converted from the PDF.
    # This is the INPUT for our analyzer.
    INPUT_IMAGE_DIR = os.path.join(project_root, 'data', 'outputs', 't58700en_converted')

    # Directory where the annotated images and JSON files will be saved.
    # This is the OUTPUT of our analyzer.
    OUTPUT_ANNOTATED_DIR = os.path.join(project_root, 'data', 'outputs', 't587000en_annotated')
    
    # Path to the pre-trained YOLO model.
    MODEL_PATH = os.path.join(project_root, 'src', 'analyzers', 'yolov10s_best.pt')

    # --- Execution ---
    # 1. Initialize the analyzer with the model path.
    analyzer = LayoutAnalyzer(model_path=MODEL_PATH)
    
    # 2. Run the batch analysis.
    analyzer.analyze_batch(input_dir=INPUT_IMAGE_DIR, output_dir=OUTPUT_ANNOTATED_DIR)
