"""
PDF loading and OCR module for pdfkg using PaddleOCR.
"""
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import paddle 
from typing import List

# Initialize PaddleOCR: automatically use GPU if available
# paddle.is_compiled_with_cuda() returns True if installed PaddlePaddle GPU version
device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
ocr_engine = PaddleOCR(lang="en|german", use_angle_cls=True, device=device)
ocr_engine = PaddleOCR(lang="en|german", use_angle_cls=True, device="cpu")

# Initialize PaddleOCR supporting English and German
# ocr_engine = PaddleOCR(lang="en|german", use_angle_cls=True, use_gpu=False)

def pdf_to_images(pdf_path: str, dpi: int = 200) -> List:
    """
    Convert each page of the PDF to a PIL image.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): Resolution for conversion.

    Returns:
        List[PIL.Image]: List of page images.
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    #for idx, img in enumerate(images, start=1):
        #img.save(f"data/outputs/page_{idx}.png")
    #return images
    return convert_from_path(pdf_path, dpi=dpi)


def ocr_images(images: List) -> List[str]:
    """
    Perform OCR on a list of PIL images and return text per page.

    Args:
        images (List[PIL.Image]): List of images to OCR.

    Returns:
        List[str]: Extracted text for each image.
    """
    text_pages: List[str] = []
    for image in images:
        ocr_results = ocr_engine.predict(image)
        # ocr_results is a list of lists: each inner list contains tuples ([box, (text, confidence)])
        # Flatten and extract text
        lines: List[str] = []
        for line in ocr_results:
            for segment in line:
                recognized_text = segment[-1][0]
                lines.append(recognized_text)
        page_text = "\n".join(lines)
        text_pages.append(page_text)
    return text_pages


def load_pdf(pdf_path: str) -> List[str]:
    """
    Load a PDF, convert pages to images, perform OCR, and return full text per page.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[str]: List of OCR text for each page.
    """
    images = pdf_to_images(pdf_path)
    return ocr_images(images)

__all__ = ["pdf_to_images", "ocr_images", "load_pdf"]
