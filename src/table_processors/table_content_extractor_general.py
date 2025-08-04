import json
import logging
import os
from typing import Any, Dict, List, Optional

import pdfplumber
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Pydantic 模型 (保持不变) ---
class KeyInsight(BaseModel):
    insight_type: str = Field(description="The category of the insight. Examples: 'Performance Data', 'Operating Limits', 'Material Specification', 'Component Compatibility'.")
    subject: str = Field(description="The primary component, model, or substance this insight pertains to.")
    description: str = Field(description="A clear, natural language sentence summarizing the key finding.")
    data_points: List[Dict[str, Any]] = Field(description="A list of key-value data that directly supports the insight.")

class ChemicalTableAnalysis(BaseModel):
    table_summary: str = Field(description="A concise, 1-2 sentence summary of the table's purpose and content.")
    key_insights: List[KeyInsight] = Field(description="A list of structured key insights extracted from the table.")
    extraction_confidence: float = Field(description="The model's confidence in the accuracy of the extraction, from 0.0 to 1.0.")


# --- 1. 核心修改: 提取为带坐标的单元格列表JSON ---
def extract_table_to_coordinate_json(pdf_path: str, page_number: int, table_index: int = 0) -> Optional[str]:
    """
    Extracts a table from a PDF and converts it into a generic list of cell objects,
    each with row, column, and text information. This format is universal for any grid-like table.

    Args:
        pdf_path: Path to the PDF file.
        page_number: The page number to extract from (1-based).
        table_index: The index of the table on the page (0-based).

    Returns:
        A JSON string representing the list of cells, or None if extraction fails.
    """
    logger.info(f"Extracting table {table_index} from page {page_number} of '{pdf_path}' to COORDINATE JSON...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
            if not tables or table_index >= len(tables):
                logger.warning(f"No table found or index out of bounds on page {page_number}.")
                return None

            raw_table = tables[table_index]
            
            cell_list = []
            for r_idx, row in enumerate(raw_table):
                for c_idx, cell_text in enumerate(row):
                    # 为每个单元格创建一个包含坐标和文本的对象
                    cell_list.append({
                        "row": r_idx,
                        "col": c_idx,
                        "text": str(cell_text or '').replace('\n', ' ').strip()
                    })

            return json.dumps(cell_list, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Failed to extract table to coordinate JSON: {e}", exc_info=True)
        return None


class TableContentExtractor:
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.analyzer = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.05,
            max_tokens=4096
        ).with_structured_output(ChemicalTableAnalysis)

    # --- 2. 核心修改: 为分析坐标JSON设计的Prompt ---
    def _create_coordinate_analysis_prompt(self) -> str:
        """
        Creates a prompt for analyzing the generic, coordinate-based JSON format.
        """
        return """
You are an expert chemical engineer and a master data analyst. Your task is to analyze the following JSON data, which is a flattened list of cells from a technical table. Each cell object has `row`, `col`, and `text` properties. You must first mentally reconstruct the table's structure using these coordinates and then provide a structured analysis.

**YOUR SOURCE OF TRUTH IS THE PROVIDED JSON CELL LIST ONLY.**

**ANALYSIS WORKFLOW:**

**STEP 1: MENTALLY RECONSTRUCT THE TABLE STRUCTURE**
- Identify the main column headers. These are typically the cells in `row: 0` or `row: 1`, starting from `col: 1`.
- Identify the main row headers. These are typically the cells in `col: 0`.
- Identify logical sections. A row where `col: 0` has text but other columns are empty often indicates a section title.

**STEP 2: GENERATE THE FINAL JSON ANALYSIS**
Based on your reconstructed understanding, populate the `ChemicalTableAnalysis` schema.

**CRITICAL REQUIREMENT: The `data_points` field is MANDATORY for every insight.**
- EVERY insight object you generate MUST contain a `data_points` field.
- If an insight is purely descriptive and has no specific data points, you MUST provide an empty list: `"data_points": []`.
- DO NOT omit the `data_points` field under any circumstances.

**HOW TO INTERPRET THE COORDINATE DATA:**

* **To find a cell's column header:** If a cell is at `(row: R, col: C)`, find the cell at `(row: 0, col: C)` or `(row: 1, col: C)` to get its column header text.
* **To create "Component Compatibility" insights:**
    * Find cells in `col: 0` whose `text` starts with "DN" (e.g., at `row: R`). This is your subject.
    * Then, look at all other cells in the same `row: R`. If a cell at `(row: R, col: C)` has `text: '●'`, find the corresponding column header at `(row: 0, col: C)`. This header is the compatible Kvs value.
    * **Subject**: e.g., "Valve DN 15"
    * **Data Points**: `[{"Available_Kvs": "0.1"}, {"Available_Kvs": "0.16"}, ...]`

* **To create "Operating Limits" insights:**
    * Find a cell in `col: 0` that describes a model or configuration (e.g., "Without balanced plug - 3374-15" at `row: R`). This is your subject.
    * Then, for each data cell in that same `row: R` (at `col: C`), find its corresponding column header at `(row: 0, col: C)`.
    * **Subject**: e.g., "Without balanced plug - 3374-15"
    * **Data Points**: `[{"Kvs": "100", "Max_Delta_P_bar": "2.0"}, {"Kvs": "80", "Max_Delta_P_bar": "3.4"}, ...]`

Your final output must be ONLY the JSON object conforming to the `ChemicalTableAnalysis` schema.
"""

    def analyze_coordinate_json(self, table_json_string: str) -> Optional[ChemicalTableAnalysis]:
        logger.info("Starting analysis of the provided coordinate-based JSON...")
        prompt = self._create_coordinate_analysis_prompt()
        message_content = f"Please analyze the following JSON cell list representing a table:\n\n{table_json_string}"
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "text", "text": message_content}
        ])
        try:
            logger.info("Sending request to GPT-4o for semantic analysis...")
            result: ChemicalTableAnalysis = self.analyzer.invoke([message])
            logger.info(f"Successfully received analysis with confidence: {result.extraction_confidence:.2f}")
            return result
        except Exception as e:
            logger.error(f"An error occurred during LLM analysis: {e}")
            return None


# --- 3. 更新后的使用示例 ---
def example_usage():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "sk-...")
    if "sk-..." in api_key:
        print("!!! WARNING: Please set your OpenAI API key before running. !!!")
        return

    pdf_path = "data/inputs/t58700en.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'.")
        return

        # --- 新增: 定义输出目录 ---
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True) # 确保目录存在

    print("="*20 + " STEP 1: Extracting Table to COORDINATE JSON " + "="*20)
    json_content = extract_table_to_coordinate_json(pdf_path, page_number=2, table_index=1)

    if not json_content:
        print("Failed to extract coordinate JSON from PDF. Aborting.")
        return

    # --- 新增: 保存中间JSON文件 ---
    intermediate_json_path = os.path.join(output_dir, "intermediate_coordinates.json")
    try:
        with open(intermediate_json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        print(f"Successfully saved intermediate JSON to: {intermediate_json_path}")
    except Exception as e:
        print(f"Error saving intermediate JSON file: {e}")
    

    print("\nSuccessfully extracted COORDINATE-BASED JSON (preview of first 10 cells):")
    print("-" * 70)
    parsed_json = json.loads(json_content)
    print(json.dumps(parsed_json[:10], indent=2, ensure_ascii=False))
    print("...")
    print("-" * 70)

    print("="*20 + " STEP 1: Extracting Table to COORDINATE JSON " + "="*20)
    json_content = extract_table_to_coordinate_json(pdf_path, page_number=2, table_index=1)

    if not json_content:
        print("Failed to extract coordinate JSON from PDF. Aborting.")
        return

    print("Successfully extracted COORDINATE-BASED JSON (preview of first 10 cells):")
    print("-" * 70)
    parsed_json = json.loads(json_content)
    print(json.dumps(parsed_json[:10], indent=2, ensure_ascii=False))
    print("...")
    print("-" * 70)

    print("\n" + "="*20 + " STEP 2: Analyzing Coordinate JSON with LLM " + "="*20)
    extractor = TableContentExtractor(api_key=api_key)
    analysis_result = extractor.analyze_coordinate_json(json_content)

    if analysis_result:
        print("\n\n" + "="*25 + " ANALYSIS COMPLETE " + "="*25)
        print("\n--- 1. TABLE SUMMARY ---")
        print(analysis_result.table_summary)
        print(f"(Extraction Confidence: {analysis_result.extraction_confidence:.2%})")
        print("\n--- 2. KEY INSIGHTS ---")
        for i, insight in enumerate(analysis_result.key_insights, 1):
            print(f"\n[INSIGHT {i}: {insight.insight_type}]")
            print(f"  - Subject: {insight.subject}")
            print(f"  - Description: {insight.description}")
            print(f"  - Supporting Data:")
            for dp in insight.data_points:
                print(f"    - {json.dumps(dp, ensure_ascii=False)}")
        print("\n" + "="*69)
    else:
        print("\n--- ANALYSIS FAILED ---")

if __name__ == "__main__":
    example_usage()
