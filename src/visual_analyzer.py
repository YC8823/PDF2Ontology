"""
Visual layout analysis of document pages using GPT-4V for region detection
and PIL for visualization with structured output validation.
"""
from dotenv import load_dotenv
load_dotenv()

import base64
import json
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, field_validator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Define Pydantic model for structured output
class DocumentRegion(BaseModel):
    """Definition of a document region"""
    region_type: str = Field(
        ...,
        description="Region type, must be one of: text, table, image, header, footer",
        alias="type"
    )
    bbox: List[float] = Field(
        ...,
        description="Normalized coordinates [x0, y0, x1, y1]",
        min_items=4,
        max_items=4
    )

    @field_validator("bbox")
    def validate_bbox(cls, v):
        if any(coord < 0 or coord > 1 for coord in v):
            raise ValueError("All coordinates must be between 0 and 1")
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError("Invalid bounding box dimensions")
        return v

# Initialize GPT-4V with structured output
vision_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_retries=3
).with_structured_output(
    schema=DocumentRegion,
    method="json_mode",
    include_raw=True
)

def get_layout_system_prompt() -> str:
    return (
        "Analyze the provided document page image and output **only** valid JSON.\n"
        "Your answer must be a top-level JSON **array** of objects, each with:\n\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "type": "text",\n'
        '    "bbox": [x0, y0, x1, y1]\n'
        "  }\n"
        "]\n"
        "```\n"
        "- **type**: one of “text”, “table”, “image”, “header”, “footer”.\n"
        "- **bbox**: four floats normalized between 0.0 and 1.0.\n"
        "Do not include any other text or markdown."
    )

def detect_layout_regions(image_path: str) -> List[Dict]:
    """
    Detect layout regions using GPT-4V with validated structured output.

    Args:
        image_path (str): Path to the input image file

    Returns:
        List[Dict]: List of region dictionaries with 'type' and 'bbox'
    """
    # Read and encode image
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=get_layout_system_prompt()),
        HumanMessage(content=[
            {"type": "text", "text": "Analyze this document page:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "auto"
            }}
        ])
    ])

    try:
        # Create processing chain
        chain = prompt_template | vision_model
        response = chain.invoke({})

        # check the type of response
        print(">>> LLM returned:", repr(response))
        print(">>> Types:", [type(r) for r in response] if isinstance(response, list) else type(response))

        # Convert Pydantic objects to dictionaries
        return [region.dict(by_alias=True) for region in response]
    
    except Exception as e:
        # If it's an OUTPUT_PARSING_FAILURE, dump the raw text
        raw = getattr(e, "raw_output", None)
        print("❗️ Raw LLM output was:\n", raw)
        raise RuntimeError(f"Layout detection failed: {str(e)}") from e

def visualize_layout(image_path: str, regions: List[Dict], output_path: str) -> None:
    """
    Draw bounding boxes and labels on the image.

    Args:
        image_path (str): Path to source image
        regions (List[Dict]): Detected regions data
        output_path (str): Path to save annotated image
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for region in regions:
        # Convert normalized coordinates
        x0, y0, x1, y1 = [coord * dim for coord, dim in zip(region["bbox"], [width, height, width, height])]
        
        # Draw bounding box
        draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
        
        # Draw label
        label = f"{region['type']}"
        draw.text((x0, y0 - 15), label, font=font, fill="red")

    img.save(output_path)

__all__ = ["detect_layout_regions", "visualize_layout"]