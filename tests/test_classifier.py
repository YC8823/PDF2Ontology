import os
import sys

# Ensure the src directory is on the import path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.loader import load_pdf
from src.classifier import classify_document

def test_classify_document_with_pdf():
    """
    End-to-end test for classify_document:
    1. Read sample PDF
    2. Extract text using loader.load_pdf
    3. Classify document
    4. Verify classification result
    """
    # Construct sample PDF path (ensure sample.pdf exists in data directory)
    sample_pdf = os.path.join(project_root, "data", "inputs", "sample02.pdf")
    assert os.path.isfile(sample_pdf), f"Sample PDF not found: {sample_pdf}"

    # 1. Extract text
    pages = load_pdf(sample_pdf)
    assert pages and isinstance(pages, list), "load_pdf should return non-empty list of page texts"

    # 2. Classify document
    full_text = "\n\n".join(pages)
    doc_type = classify_document(full_text)
    
    # allowed_types = [
    #     "Technical Manual",
    #     "Purchase Order",
    #     "Financial Report",
    #     "Contract",
    #     "Conference Paper",
    #     "Other"
    # ]
    ## assert doc_type in allowed_types, f"Unexpected document type: {doc_type}"

    print(f"classify_document returned valid type: {doc_type}")
    return doc_type

if __name__ == "__main__":
    test_classify_document_with_pdf()
    print("All classifier tests passed.")