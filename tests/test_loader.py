"""
Test script for loader.py to verify PDF to image conversion and OCR functionality.
"""
import os
import sys

# Ensure project root is in Python path to import src module
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.loader import pdf_to_images, ocr_images, load_pdf


def test_pdf_to_images():
    # Path to sample PDF
    sample_pdf = os.path.join(project_root, "data/inputs/sample02.pdf")
    assert os.path.exists(sample_pdf), f"Sample PDF not found at {sample_pdf}"

    # Convert PDF to images
    page_images = pdf_to_images(sample_pdf)
    assert isinstance(page_images, list), "pdf_to_images should return a list"
    assert page_images, "pdf_to_images returned an empty list"
    print(f"pdf_to_images: converted to {len(page_images)} page image(s)")


def test_ocr_images():
    # Use the first page image for OCR test
    sample_pdf = os.path.join(project_root, "data/inputs/sample02.pdf")
    page_images = pdf_to_images(sample_pdf)
    first_page = page_images[0]

    # Perform OCR
    ocr_texts = ocr_images([first_page])
    assert isinstance(ocr_texts, list), "ocr_images should return a list"
    assert ocr_texts and isinstance(ocr_texts[0], str), "OCR result should be a list of strings"
    print("ocr_images: extracted text sample:\n", ocr_texts[0][:200], "...")


def test_load_pdf():
    # Combined loader test
    sample_pdf = os.path.join(project_root, "data/inputs/sample02.pdf")
    texts = load_pdf(sample_pdf)
    assert isinstance(texts, list), "load_pdf should return a list"
    assert texts and isinstance(texts[0], str), "load_pdf should return list of strings"
    print(f"load_pdf: obtained {len(texts)} page(s) of text")


if __name__ == '__main__':
    test_pdf_to_images()
    test_ocr_images()
    test_load_pdf()
    print("Loader tests passed.")
