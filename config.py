import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '3'))
    DEFAULT_OUTPUT_DIR = os.getenv('DEFAULT_OUTPUT_DIR', './output/')
    
    # Table processing settings
    TABLE_CONFIDENCE_THRESHOLD = 0.7
    TABLE_OVERLAP_THRESHOLD = 0.5
    
    # CV processing settings
    CV_LINE_KERNEL_SIZE = 40
    CV_LINE_THRESHOLD = 0.3
    
    # Ontology settings
    MAX_ENTITIES_PER_EXTRACTION = 100
    MAX_TRIPLETS_PER_DOCUMENT = 200