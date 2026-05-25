# src/knowledge_extractor/one_shot_triple_extractor.py
"""
One-Shot Triple Extractor (Improved)

Extracts OWL/RDFS triples from PDF datasheets in a single LLM conversation,
using ontology schema as guidance.

Improvements:
1. Dynamic generation of ALL object properties and datatype properties
2. Explicit inverse property handling
3. Namespace-aware TTL generation for seamless integration
4. Post-processing for triple validation
"""

import os
import sys
import json
import base64
import logging
import re
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Dynamic path setup
try:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    project_root = Path.cwd()

from src.preprocessors.ontology_loader import OntologyLoaderOwlready2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================
# === ENHANCED PROMPT TEMPLATE
# =====================================================

EXTRACTION_PROMPT_TEMPLATE = """
## Your Role
You are a Process Industry Knowledge Engineer responsible for extracting structured information from device datasheet PDFs and generating OWL/RDFS triples according to a given ontology schema.

This task adopts an **"Incremental Correction" strategy**:
- You will progressively form an understanding of device categories and structures as you read different parts of the PDF.
- When you discover new evidence in subsequent content (tables, diagrams, etc.) that shows your previous understanding was incomplete or incorrect, you **must explicitly point out and correct** your earlier conclusions.
- The final output should contain **only the latest, corrected, and consistent version** of triples (do not output old, erroneous versions).

---

## Input 1: Ontology Schema (YAML Format)

This is the TBox definition of the ontology, including classes, object properties, data properties, and their domains/ranges.

```yaml
{ontology_yaml}
```

---

## CRITICAL: Complete Property Usage Requirements

### All Available Object Properties (USE ALL APPLICABLE):
{all_object_properties}

### All Available Datatype Properties (USE ALL APPLICABLE):
{all_datatype_properties}

### Inverse Property Pairs (CREATE BOTH DIRECTIONS):
{inverse_property_pairs}

**MANDATORY RULES:**
1. **Use ALL applicable properties** from the lists above when creating instances
2. **Do NOT invent** property names that don't exist in the schema above
3. **Always create inverse triples** when the relationship has an inverse property defined
   - Example: If you write `:Device001 :hasComponent :Component001`
   - You MUST also write `:Component001 :isComponentOf :Device001` (if inverse pair exists)
4. **Respect domain and range constraints** - only use properties where instance types match domains/ranges
5. **Use consistent namespace prefix ':'** for all instances (will be replaced with actual IRI during post-processing)

**Important**: The schema above is the **sole authoritative source**. You may add new classes (as subclasses) or new individuals when justified, but NEVER invent properties not listed above.

---

## Input 2: PDF Content (Images)

The PDF datasheet has been converted to images, which are provided after this prompt. Each image represents one page of the datasheet.

The datasheet typically contains:
1. **Textual descriptions** (device names, component relationships)
2. **Tables** with dimensional data (pay special attention to merged cells and column/row headers), 
3. **Technical drawings** showing:
   - Horizontal dimensions (L, L1, a, b, c, etc.)
   - Vertical dimensions (H, H1, H2, H9, etc.)
   - Diameters (often marked with ø symbol)
   - Dimension lines with arrows and measurements

**Critical for Table Identification and Filtering**:
When analyzing tables, you must distinguish between:

1. **DIMENSIONAL TABLES (TO EXTRACT):**
   - Tables where dimension parameters (L, H, B, D, etc.) are the PRIMARY subject
   - Column/row headers explicitly label dimension names (e.g., "H1", "L2", "Diameter D")
   - Values in cells represent physical measurements (typically in mm, cm, inch)
   - Example: "Dimensions for DN15-DN50" table with columns [DN, L, H, B, Weight]

2. **NON-DIMENSIONAL TABLES (TO IGNORE):**
   - Tables where dimension values appear ONLY in headers/row labels but NOT as the primary data
   - Tables focusing on performance characteristics (Kv, Cv, flow rate, pressure, temperature ranges)
   - Example: "Kv values for different port diameters" table with structure:
     * Column headers: DN15 (d=10mm), DN20 (d=15mm), DN25 (d=20mm)
     * Row data: Kv values
     * ❌ DO NOT EXTRACT - diameter info is metadata, not dimensional data

**Decision Framework:**
- Ask: "What is this table ABOUT?" 
  - If answer is "device dimensions/measurements" → EXTRACT
  - If answer is "performance data/specifications" → IGNORE
- Check: "Are dimension values in the DATA cells or only in HEADERS?"
  - In data cells → EXTRACT
  - Only in headers (as categorical labels) → IGNORE
   
**Critical for Technical Drawings**:
- Carefully identify dimension line orientations:
  - Lines predominantly left-right → "horizontal"
  - Lines predominantly up-down → "vertical"
  - Circular measurements or ø symbol → "diameter"
- Do NOT assume orientation from parameter name alone (e.g., "H" is not always vertical)
- Look for the actual dimension lines and their visual direction

---

## Analysis and Modeling Requirements

### 1. Provide Initial Modeling Based on Current Information
- Start by building a preliminary understanding based on what you've seen so far.

### 2. Correct When Discovering New Information
- As you read subsequent pages, if you discover new structural variant information, you **must perform "correction"**.
- When correcting, you should:
  a) **Explicitly state**: "My previous conclusion about 'XXX' is no longer accurate."
  b) **Provide correction strategy**: Explain what changed and why.
  c) **Update your modeling**: Revise subclasses and instances to align with the new structural division.

### 3. Use Schema to Constrain Triples
- All triples must comply with the domains and ranges declared in the schema.
- Use ALL applicable properties from the schema for each instance.

### 4. Create Dimension Instances
**Core Extraction Focus:**
The primary goal is to establish clear mappings:
- **Model/Part Number** (e.g., "Type 3241", "Series XYZ")
- **Variant Options** (e.g., DN size, PN rating, material code, special features)
- **Dimensional Parameters** (e.g., L, H, B, D, Weight)

**Extraction Protocol:**
For each **(Model + Variant Combination, Parameter Name)** found in DIMENSIONAL TABLES:
- Create a `Dimension` instance with format: `:Dim_Model_Variant_ParameterName`
- Example: `:Dim_Type3241_DN25_PN40_HeatingJacket_H9`

**Variant Identification Priority:**
1. **Nominal Size (DN/NPS)**: Most common variant, always extract if present
2. **Pressure Rating (PN/Class)**: Extract when specified
3. **Functional Variants**: Heating jacket, insulation, bellows, extended spindle, etc.
4. **Material/Construction**: If it affects dimensions (e.g., lined vs. unlined)
5. **Connection Type**: Flange type, thread type (if dimensionally relevant)

**Linking Protocol:**
- Use ALL dimension-related datatype properties from the schema
- Link `Dimension` instances to their device/component instances via `:hasDimension` property
- Ensure each dimension instance clearly indicates its variant context through naming and properties

---

## Naming Conventions

### For Individuals (Instances):
Use readable prefixes:
- `:DeviceModel_XXX_DN15` for device instances
- `:Component_Type_Variant_DN15` for component instances  
- `:Dim_ModelXXX_DN15_ParameterName` for Dimension instances

### For hasModelName:
Write the original model string from the PDF, e.g.:
- `"Type 3241 Valve DN 15 with heating jacket"`

---

## Output Format

### 1. Extraction Report (Natural Language)
- Provide a brief explanation of your analysis process.
- **If corrections were made**, explain:
  - What your initial understanding was
  - What new evidence led to the correction
  - How you revised the modeling
- Summarize the final structure (classes, instances, key relationships)

### 2. Turtle Triples (OWL/RDFS)
- Output **only the final, corrected, and consistent** version of triples.
- Do **not** output intermediate erroneous triples.
- Use Turtle syntax (`.ttl` format)
- Use prefix `:` for all instances (will be automatically replaced with correct namespace)

**Example Triples** (illustrative only):

```turtle
:Insulated3241ValveBody rdf:type :ValveBody .
:Insulated3241ValveBody :hasModelName "Type 3241 Valve with insulating section or bellows" .

:VB_3241_DN15_Insulated rdf:type :Insulated3241ValveBody ;
    :hasNominalSizeDN 15 ;
    :hasDimension :Dim_VB3241_DN15_H9_short .

:Dim_VB3241_DN15_H9_short rdf:type :Dimension ;
    :hasDimensionParameterName "H9_short_with_bellows" ;
    :hasDimensionType "vertical" ;
    :hasDimensionUnit "mm" ;
    :hasDimensionValue 409.0 .
```

---

## Critical Behavior Guidelines

### 1. Encourage Mid-Process Corrections
- When you discover new evidence in later pages/tables/diagrams that shows your earlier class divisions or instance associations were unreasonable, you **must** explicitly mark a "correction point" in your natural language explanation.
- Example: "Previously I assumed X, but based on Table A/B, I now revise to Y."

### 2. Final Triples Must Be Consistent
- Do **not** output two contradictory versions of triples.
- The correction process should only appear in the natural language explanation section.
- The RDF/OWL section should be clean and conflict-free.

### 3. Complete Property Coverage
- Review the property lists above and ensure you've used all applicable properties for each instance type.
- Don't forget inverse properties - if you create A→B, also create B→A when inverse exists.

### 4. Completeness Protocol 
- **Exhaustive Extraction:** You must process **EVERY** applicable row in the tables and **EVERY** parameter mentioned.
- **No "Snippets":** Do not output comments like "// ... remaining rows omitted for brevity." You must output the full Turtle code for all instances.
- **Full Coverage:** If a table lists 20 different DN sizes, you must generate triples for all 20 sizes, and go through all pages and tables in the PDF.
- do not skip any dimension parameters or variant combinations, do not save tokens by omitting data.
---

## Task Summary

Based on the above requirements, analyze the provided PDF images and ontology schema:
1. Perform step-by-step modeling as you examine each page
2. Correct as needed when new information is discovered
3. Use ALL applicable properties from the schema
4. Create inverse triples for all bidirectional relationships
5. Output the **final corrected OWL/RDFS triples** in Turtle format

Please structure your response as follows:

```
## EXTRACTION REPORT

[Your natural language analysis and correction explanations here]

## OWL/RDFS TRIPLES

```turtle
[Your final triples in Turtle syntax here]
```
```

Now, please analyze the datasheet images provided below:
"""


# =====================================================
# === PDF TO IMAGE CONVERTER
# =====================================================

def convert_pdf_to_images(
    pdf_path: str, 
    max_pages: Optional[int] = None,
    dpi: int = 300
) -> List[Tuple[Any, int]]:
    """
    Convert PDF pages to images for multimodal LLM processing.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to convert (None = all)
        dpi: Resolution for rendering (default: 300)
        
    Returns:
        List of (PIL.Image, page_number) tuples
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError:
        logger.error("Required libraries not installed. Install with: pip install PyMuPDF Pillow")
        return []
    
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        num_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
        
        logger.info(f"Converting {num_pages} pages to images (DPI: {dpi})...")
        
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            
            # Render page to pixmap at specified DPI
            zoom = dpi / 72  # 72 is default DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            if pix.alpha:
                img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
                img = img.convert("RGB")  # Remove alpha channel
            else:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            images.append((img, page_num + 1))  # 1-indexed page numbers
            logger.debug(f"  Page {page_num + 1}: {img.size[0]}x{img.size[1]} pixels")
        
        doc.close()
        
        logger.info(f"✓ Converted {len(images)} pages to images")
        return images
        
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        return []


def encode_image_to_base64(image: Any) -> str:
    """
    Encode PIL Image to Base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string
    """
    import io
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# =====================================================
# === IMPROVED EXTRACTOR CLASS
# =====================================================

class OneShotTripleExtractor:
    """
    One-shot triple extraction from PDF datasheets using LLM.
    
    Improvements:
    - Dynamic property list generation
    - Namespace-aware output
    - Post-processing for integration
    """
    
    def __init__(
        self,
        ontology_path: str,
        api_key: str,
        model_name: str = "gpt-5.1"
    ):
        """
        Initialize extractor.
        
        Args:
            ontology_path: Path to ontology file (.rdf, .owl, .ttl)
            api_key: OpenAI API key
            model_name: LLM model to use
        """
        logger.info("="*70)
        logger.info("Initializing Enhanced One-Shot Triple Extractor")
        logger.info("="*70)
        
        # Validate ontology path
        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
        
        self.ontology_path = ontology_path
        
        # Load and serialize ontology
        logger.info(f"Loading ontology: {ontology_path}")
        self.ontology_loader = OntologyLoaderOwlready2(ontology_path)
        
        if not self.ontology_loader.load():
            raise RuntimeError(f"Failed to load ontology from {ontology_path}")
        
        if not self.ontology_loader.extract_structure():
            raise RuntimeError(f"Failed to extract ontology structure")
        
        # Get YAML representation
        self.ontology_yaml = self.ontology_loader.export_to_yaml()
        logger.info(f"✓ Ontology loaded and serialized to YAML")
        
        # Store namespace for post-processing
        self.base_namespace = self.ontology_loader.loos.base_namespace
        logger.info(f"✓ Base namespace: {self.base_namespace}")
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {model_name}")
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=20000
        )
        logger.info("✓ LLM initialized\n")
    
    def _format_properties_list(
        self, 
        properties: Dict[str, Any], 
        property_type: str
    ) -> str:
        """
        Format properties into a readable list for the prompt.
        
        Args:
            properties: Dictionary of PropertyInfo objects
            property_type: "object" or "datatype"
            
        Returns:
            Formatted string
        """
        if not properties:
            return f"No {property_type} properties defined in ontology."
        
        lines = []
        for prop_name, prop_info in properties.items():
            domain_str = ", ".join(prop_info.domain) if prop_info.domain else "Thing"
            range_str = ", ".join(prop_info.range) if prop_info.range else "Thing"
            
            line = f"- **{prop_name}**: domain=[{domain_str}], range=[{range_str}]"
            
            if prop_info.label:
                line += f" | label=\"{prop_info.label}\""
            
            if prop_info.inverse_of and property_type == "object":
                line += f" | inverse_of={prop_info.inverse_of}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_inverse_pairs(self, inverse_pairs: List[Tuple[str, str]]) -> str:
        """
        Format inverse property pairs for the prompt.
        
        Args:
            inverse_pairs: List of (prop1, prop2) tuples
            
        Returns:
            Formatted string
        """
        if not inverse_pairs:
            return "No inverse property pairs defined in ontology."
        
        lines = []
        for prop1, prop2 in inverse_pairs:
            lines.append(f"- **{prop1}** ↔ **{prop2}**")
        
        return "\n".join(lines)
    
    def _build_prompt(self) -> str:
        """
        Build extraction prompt with dynamic property lists.
        
        Returns:
            Formatted prompt string
        """
        # Get structured info from LOOS
        loos = self.ontology_loader.loos
        
        # Format object properties
        obj_props_str = self._format_properties_list(
            loos.object_properties, 
            "object"
        )
        
        # Format datatype properties
        data_props_str = self._format_properties_list(
            loos.datatype_properties, 
            "datatype"
        )
        
        # Format inverse pairs
        inverse_pairs_str = self._format_inverse_pairs(
            loos.inverse_property_pairs
        )
        
        # Build complete prompt
        return EXTRACTION_PROMPT_TEMPLATE.format(
            ontology_yaml=self.ontology_yaml,
            all_object_properties=obj_props_str,
            all_datatype_properties=data_props_str,
            inverse_property_pairs=inverse_pairs_str
        )
    
    def _parse_llm_response(self, response_text: str) -> Tuple[str, str]:
        """
        Parse LLM response into report and triples.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            (report, triples) tuple
        """
        # Try to find triples section
        turtle_pattern = r"```turtle\s*(.*?)\s*```"
        matches = re.findall(turtle_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            triples = matches[-1]  # Take last match (final version)
        else:
            # Fallback: try to find any code block
            code_pattern = r"```\s*(.*?)\s*```"
            matches = re.findall(code_pattern, response_text, re.DOTALL)
            triples = matches[-1] if matches else ""
        
        # Extract report (everything before triples)
        if triples:
            report = response_text.split("```")[0].strip()
        else:
            report = response_text
        
        return report, triples
    
    def _post_process_triples(self, raw_triples: str) -> str:
        """
        Post-process extracted triples for integration.
        
        Operations:
        1. Replace placeholder prefix ':' with actual namespace
        2. Remove any LLM-generated prefix declarations (will be added by integration tool)
        3. Validate basic syntax
        
        Args:
            raw_triples: Raw triples from LLM
            
        Returns:
            Processed triples ready for integration
        """
        logger.info("Post-processing extracted triples...")
        
        # Remove existing prefix declarations (will be added during integration)
        lines = raw_triples.split('\n')
        processed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip prefix declarations and comments about prefixes
            if stripped.startswith('@prefix') or stripped.startswith('# Auto-generated'):
                continue
            
            # Skip empty lines at the start
            if not processed_lines and not stripped:
                continue
            
            processed_lines.append(line)
        
        processed_triples = '\n'.join(processed_lines).strip()
        
        # Basic validation: check for common patterns
        if ':' not in processed_triples:
            logger.warning("No instances found in triples (missing ':' prefix)")
        
        if 'rdf:type' not in processed_triples and 'a ' not in processed_triples:
            logger.warning("No type declarations found in triples")
        
        logger.info(f"✓ Post-processing complete ({len(processed_lines)} lines)")
        
        return processed_triples
    
    def extract_from_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        max_pages: Optional[int] = None,
        dpi: int = 300
    ) -> bool:
        """
        Extract triples from PDF datasheet using multimodal vision analysis.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for results
            max_pages: Maximum pages to process (None = all)
            dpi: Resolution for PDF rendering (default: 300)
            
        Returns:
            Success status
        """
        logger.info("="*70)
        logger.info("Starting Enhanced One-Shot Triple Extraction")
        logger.info("="*70)
        logger.info(f"PDF: {pdf_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"DPI: {dpi}")
        
        # Validate PDF
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images
        logger.info("\n[Step 1/4] Converting PDF to images...")
        images = convert_pdf_to_images(pdf_path, max_pages=max_pages, dpi=dpi)
        
        if not images:
            logger.error("Failed to convert PDF to images")
            return False
        
        logger.info(f"✓ Converted {len(images)} pages to images")
        
        # Check if too many pages
        if len(images) > 20:
            logger.warning(
                f"Processing {len(images)} pages may consume significant API tokens. "
                f"Consider using max_pages parameter to test with fewer pages first."
            )
        
        # Encode images to Base64
        logger.info("\n[Step 2/4] Encoding images to Base64...")
        image_messages = []
        
        for img, page_num in images:
            try:
                img_b64 = encode_image_to_base64(img)
                image_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high"
                    }
                })
                logger.debug(f"  Page {page_num}: {len(img_b64)} chars (Base64)")
            except Exception as e:
                logger.error(f"Failed to encode page {page_num}: {e}")
                return False
        
        logger.info(f"✓ Encoded {len(image_messages)} images")
        
        # Build prompt with dynamic property lists
        logger.info("\n[Step 3/4] Building enhanced extraction prompt...")
        prompt_text = self._build_prompt()
        
        logger.info(f"✓ Prompt built ({len(prompt_text)} characters)")
        logger.info(f"  - Object properties: {len(self.ontology_loader.loos.object_properties)}")
        logger.info(f"  - Datatype properties: {len(self.ontology_loader.loos.datatype_properties)}")
        logger.info(f"  - Inverse pairs: {len(self.ontology_loader.loos.inverse_property_pairs)}")
        
        # Construct multimodal message
        message_content = [
            {"type": "text", "text": prompt_text}
        ]
        message_content.extend(image_messages)
        
        total_content_items = len(message_content)
        logger.info(f"✓ Message constructed ({total_content_items} content items: 1 text + {len(image_messages)} images)")
        
        # Call LLM
        logger.info("\n[Step 4/4] Calling multimodal LLM for extraction...")
        logger.info("⚠️  This may take several minutes for complex datasheets...")
        
        try:
            messages = [
                SystemMessage(content="You are an expert knowledge engineer specializing in industrial automation ontologies. Follow all schema constraints strictly."),
                HumanMessage(content=message_content)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            logger.info(f"✓ Received response ({len(response_text)} characters)")
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            logger.error("This might be due to:")
            logger.error("  1. Token limit exceeded (try reducing max_pages or dpi)")
            logger.error("  2. API rate limit")
            logger.error("  3. Network issues")
            return False
        
        # Parse response
        logger.info("\nParsing LLM response...")
        report, raw_triples = self._parse_llm_response(response_text)
        
        # Post-process triples
        processed_triples = self._post_process_triples(raw_triples)
        
        # Save outputs
        pdf_basename = Path(pdf_path).stem
        
        # Save full response
        full_response_path = Path(output_dir) / f"{pdf_basename}_full_response.txt"
        with open(full_response_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        logger.info(f"✓ Saved full response: {full_response_path}")
        
        # Save extraction report
        report_path = Path(output_dir) / f"{pdf_basename}_extraction_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Extraction Report: {pdf_basename}\n\n")
            f.write(f"**PDF Pages Analyzed:** {len(images)}\n\n")
            f.write(f"**Resolution:** {dpi} DPI\n\n")
            f.write(f"**Base Namespace:** {self.base_namespace}\n\n")
            f.write("---\n\n")
            f.write(report)
        logger.info(f"✓ Saved extraction report: {report_path}")
        
        # Save processed triples (WITHOUT prefixes - will be added during merge)
        if processed_triples:
            triples_path = Path(output_dir) / f"{pdf_basename}_extracted_abox.ttl"
            with open(triples_path, 'w', encoding='utf-8') as f:
                f.write("# Extracted ABox instances (without prefix declarations)\n")
                f.write("# This file should be merged with TBox using merge_and_validate_ttl.py\n\n")
                f.write(processed_triples)
            logger.info(f"✓ Saved processed triples: {triples_path}")
            
            # Save metadata for integration
            metadata_path = Path(output_dir) / f"{pdf_basename}_extraction_metadata.json"
            metadata = {
                "pdf_filename": os.path.basename(pdf_path),
                "ontology_path": self.ontology_path,
                "base_namespace": self.base_namespace,
                "pages_processed": len(images),
                "dpi": dpi,
                "object_properties_count": len(self.ontology_loader.loos.object_properties),
                "datatype_properties_count": len(self.ontology_loader.loos.datatype_properties),
                "inverse_pairs_count": len(self.ontology_loader.loos.inverse_property_pairs),
                "extracted_triples_file": os.path.basename(triples_path)
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Saved extraction metadata: {metadata_path}")
            
        else:
            logger.warning("⚠️  No triples were extracted from the response")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("EXTRACTION COMPLETE")
        logger.info("="*70)
        logger.info(f"Pages processed: {len(images)}")
        logger.info(f"Report: {report_path}")
        logger.info(f"Processed triples: {triples_path if processed_triples else 'None'}")
        logger.info(f"Metadata: {metadata_path if processed_triples else 'None'}")
        logger.info(f"\nNext step: Use merge_and_validate_ttl.py to integrate into ontology")
        logger.info("="*70)
        
        return True


# =====================================================
# === MAIN EXECUTION
# =====================================================

def main():
    """Main execution function"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    # Configure paths
    try:
        project_root = Path(__file__).parent.parent.parent
    except NameError:
        project_root = Path.cwd()
    
    # Input paths
    ontology_path = project_root / "data" / "ontology" / "DeviceDimension_no_anchor.rdf"
    pdf_path = project_root / "data" / "inputs" / "t58740en.pdf"
    
    # Output directory
    output_dir = project_root / "data" / "outputs" / "one_shot_extraction"
    
    # Validate inputs
    if not ontology_path.exists():
        logger.error(f"Ontology file not found: {ontology_path}")
        logger.error("Please provide a valid ontology path")
        sys.exit(1)
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        logger.error("Please provide a valid PDF path")
        sys.exit(1)
    
    # Initialize extractor
    try:
        extractor = OneShotTripleExtractor(
            ontology_path=str(ontology_path),
            api_key=api_key,
            model_name="gpt-5.1"
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        sys.exit(1)
    
    # Extract triples
    try:
        success = extractor.extract_from_pdf(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
            max_pages=None,
            dpi=300
        )
        
        if success:
            logger.info("\n✓ Extraction successful!")
        else:
            logger.error("\n✗ Extraction failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()