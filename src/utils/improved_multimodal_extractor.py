# src/knowledge_extractor/improved_multimodal_extractor.py

import logging
import base64
import os
import sys
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 动态添加项目根目录到sys.path
try:
    project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
except Exception:
    project_root_path = ""

from src.pydantic_models.modular_analysis_models import (
    ImageContentClassification,
    SubImageIdentification,
)

logger = logging.getLogger(__name__)


class ImprovedMultimodalExtractor:
    """
    改进的多模态提取器
    - 使用归一化网格坐标系统（左上角为原点）
    - 简化输出格式，仅提取参数名称和方向
    - 与表格提取结果格式对齐
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        """初始化提取器"""
        self.base_llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=2048,
        )
        
    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """将图片文件编码为Base64字符串"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败 {image_path}: {e}")
            return None

    # =====================================================
    # 步骤1：图像内容分类
    # =====================================================
    
    def classify_image_content(self, image_b64: str) -> Optional[ImageContentClassification]:
        """步骤1：对图像进行内容分类"""
        prompt = """
You are an expert image classifier for technical documents.

Analyze this image and classify its content type:
- 'engineering_drawing': Technical drawings with dimensional annotations, schematics, blueprints
- 'product_photo': Photographs of actual products, devices, or equipment
- 'table_image': Images containing tables, charts, or structured data
- 'other': Any other type of content (text blocks, logos, etc.)

Provide a confidence score between 0 and 1, and write a brief description.

Focus on being accurate and decisive in your classification.
"""
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        )
        
        try:
            llm_with_parser = self.base_llm.with_structured_output(ImageContentClassification)
            result = llm_with_parser.invoke([message])
            logger.debug(f"图像分类完成: {result.content_type} (置信度: {result.confidence})")
            return result
        except Exception as e:
            logger.error(f"图像分类失败: {e}")
            return None

    # =====================================================
    # 步骤2：子图识别
    # =====================================================
    
    def identify_sub_images(self, image_b64: str, content_type: str) -> Optional[SubImageIdentification]:
        """步骤2：识别图像中的独立子图"""
        if content_type != "engineering_drawing":
            return SubImageIdentification(
                has_multiple_diagrams=False,
                sub_images=[{
                    "sub_image_id": "sub_1",
                    "title": None,
                    "approximate_bbox": [0.0, 0.0, 1.0, 1.0]
                }]
            )
        
        prompt = """
You are analyzing an engineering drawing to identify independent sub-diagrams.

Your task:
1. Determine if this image contains multiple independent technical diagrams or just one
2. For each sub-diagram, provide an approximate bounding box in normalized coordinates [0-1]
3. Extract any visible title or identifier for each sub-diagram (like "Type 3321-E1", "Figure 2.5")

IMPORTANT: If you cannot determine the title with certainty, return None (null) instead of guessing.

Guidelines:
- Independent sub-diagrams are separate technical drawings that could stand alone
- Each should show a distinct device, component, or view
- Ignore small detail callouts or dimension annotations as separate diagrams
- Focus on main technical illustrations

Be conservative: when in doubt, treat as a single diagram.
"""
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        )
        
        try:
            llm_with_parser = self.base_llm.with_structured_output(SubImageIdentification)
            result = llm_with_parser.invoke([message])
            logger.debug(f"子图识别完成: 发现 {len(result.sub_images)} 个子图")
            return result
        except Exception as e:
            logger.error(f"子图识别失败: {e}")
            return None

    # =====================================================
    # 步骤3：简化的尺寸组件提取（仅参数名和方向）
    # =====================================================
    
    def extract_dimension_components_simple(self, image_b64: str, sub_images: List[Dict]) -> Dict[str, Any]:
        """
        步骤3：提取尺寸组件（简化版本，仅提取参数名称和方向）
        
        返回格式与表格提取对齐：
        {
            "source_image": "...",
            "all_device_dimensions": [
                {
                    "device_model_name": "Type 3244 Valve" or None,
                    "dimensions": [
                        {"parameter_name": "H1", "orientation": "vertical"},
                        {"parameter_name": "L", "orientation": "horizontal"}
                    ]
                }
            ],
            "summary": "..."
        }
        """
        
        # 准备子图信息用于prompt
        sub_images_info = []
        for sub_img in sub_images:
            info = f"- ID: {sub_img['sub_image_id']}"
            if sub_img.get('title'):
                info += f", Title: {sub_img['title']}"
            sub_images_info.append(info)
        
        prompt = f"""
You are analyzing dimension annotations in an engineering drawing.

COORDINATE SYSTEM:
- The image is treated as a normalized grid with coordinates from [0, 1]
- Origin (0, 0) is at the TOP-LEFT corner
- X-axis extends to the right (horizontal)
- Y-axis extends downward (vertical)

Sub-images identified:
{chr(10).join(sub_images_info)}

YOUR TASK:
Extract ALL dimension parameters from this engineering drawing and classify their orientations.

For EACH sub-image:
1. Identify the device model name from the title (if available and certain)
   - If uncertain or not visible, use null
   - Examples: "Type 3244 Valve", "Type 3321-E1 Actuator"

2. Find ALL dimension annotations in that sub-image
   - Parameter names like: 'H', 'H1', 'H2', 'L', 'L1', 'ø25', 'a', 'B', 'D'
   - DO NOT fabricate parameters that are not clearly visible

3. For each parameter, classify its orientation:
   - 'vertical': Primarily up-down dimension lines
   - 'horizontal': Primarily left-right dimension lines
   - 'diameter': Parameters with 'ø' symbol or circular measurements
   - 'other': Angles, radii, or any other type

CRITICAL RULES:
- Only extract parameters that are clearly visible in the image
- If device model name is uncertain, use null (do not guess)
- Base orientation on the actual dimension line direction, not just the parameter name
- Focus on accuracy over completeness

Output format (JSON):
{{
  "source_image": "engineering_drawing",
  "all_device_dimensions": [
    {{
      "device_model_name": "Type 3244 Valve" or null,
      "dimensions": [
        {{"parameter_name": "H1", "orientation": "vertical"}},
        {{"parameter_name": "L", "orientation": "horizontal"}}
      ]
    }}
  ],
  "summary": "Extracted X parameters from Y sub-images"
}}
"""
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        )
        
        try:
            # 使用dict输出而不是structured output，以便更灵活
            result = self.base_llm.invoke([message])
            
            # 尝试解析JSON响应
            import json
            try:
                result_dict = json.loads(result.content)
                logger.info(f"简化提取完成: {result_dict.get('summary', 'No summary')}")
                return result_dict
            except json.JSONDecodeError:
                logger.error(f"无法解析LLM响应为JSON: {result.content[:200]}")
                return {
                    "source_image": "unknown",
                    "all_device_dimensions": [],
                    "summary": "Extraction failed - invalid JSON response"
                }
                
        except Exception as e:
            logger.error(f"尺寸组件提取失败: {e}")
            return {
                "source_image": "unknown",
                "all_device_dimensions": [],
                "summary": f"Extraction failed - {str(e)}"
            }

    # =====================================================
    # 主分析方法（简化版本）
    # =====================================================
    
    def analyze_image_simplified(self, image_path: str, image_title: str) -> Dict[str, Any]:
        """
        执行简化的图像分析（3步流程）
        
        Args:
            image_path: 图像文件路径
            image_title: 图像标题
            
        Returns:
            简化的提取结果字典
        """
        logger.info(f"开始简化分析: {image_title}")
        
        # 编码图像
        image_b64 = self._encode_image_to_base64(image_path)
        if not image_b64:
            return {
                "image_path": image_path,
                "image_title": image_title,
                "source_image": image_title,
                "all_device_dimensions": [],
                "summary": "Failed to load image",
                "analysis_status": "failed"
            }
        
        try:
            # 步骤1：图像分类
            logger.debug("执行步骤1: 图像分类")
            classification = self.classify_image_content(image_b64)
            if not classification:
                raise Exception("图像分类失败")
            
            # 如果不是工程图，直接返回空结果
            if classification.content_type != "engineering_drawing":
                logger.info(f"图像不是工程图: {classification.content_type}")
                return {
                    "image_path": image_path,
                    "image_title": image_title,
                    "source_image": image_title,
                    "all_device_dimensions": [],
                    "summary": f"Not an engineering drawing: {classification.content_type}",
                    "analysis_status": "skipped"
                }
            
            # 步骤2：子图识别
            logger.debug("执行步骤2: 子图识别")
            sub_images = self.identify_sub_images(image_b64, classification.content_type)
            if not sub_images:
                raise Exception("子图识别失败")
            
            # 步骤3：简化的尺寸提取
            logger.debug("执行步骤3: 简化尺寸提取")
            extraction_result = self.extract_dimension_components_simple(
                image_b64, 
                [si.dict() for si in sub_images.sub_images]
            )
            
            # 添加元信息
            extraction_result["image_path"] = image_path
            extraction_result["image_title"] = image_title
            extraction_result["analysis_status"] = "success"
            
            logger.info(f"简化分析完成: {image_title}")
            return extraction_result
            
        except Exception as e:
            logger.error(f"简化分析失败 {image_title}: {e}", exc_info=True)
            return {
                "image_path": image_path,
                "image_title": image_title,
                "source_image": image_title,
                "all_device_dimensions": [],
                "summary": f"Analysis failed: {str(e)}",
                "analysis_status": "failed"
            }


if __name__ == "__main__":
    # 测试代码
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("请设置OPENAI_API_KEY环境变量")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 测试路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_image = os.path.join(project_root, "data", "outputs", "cropped_images", "page_002_crop_00.png")
    
    if os.path.exists(test_image):
        extractor = ImprovedMultimodalExtractor(api_key=api_key)
        result = extractor.analyze_image_simplified(test_image, "Test Engineering Drawing")
        
        print("\n" + "="*60)
        print("测试结果:")
        print("="*60)
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"测试图像不存在: {test_image}")