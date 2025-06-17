# ==================== examples/table_analysis_example.py ====================
"""
Table Analysis Example
Demonstrates how to use the PDF2Ontology system for table detection and analysis
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pydantic_models.enums import RegionType
from src.pydantic_models.region_models import DocumentRegion, DocumentLayout
from src.core.document_analyzer import DocumentAnalyzer
from src.core.region_detector import RegionDetector
from src.utils.image_utils import convert_pdf_to_images


def create_realistic_table_test():
    """
    Create a realistic test document with complex table structure
    This test doesn't require any external files
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("💡 Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("🎨 Creating realistic table test document...")
    
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a document-like image
    img_width, img_height = 1200, 800
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to get a better font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 18)
            font_medium = ImageFont.truetype("arial.ttf", 14) 
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font_large = font_medium = font_small = ImageFont.load_default()
    
    # Document header
    draw.text((50, 30), "TECHNICAL SPECIFICATION DATASHEET", fill='black', font=font_large)
    draw.text((50, 60), "Model: XYZ-2024 | Version: 2.1", fill='gray', font=font_medium)
    
    # Introduction paragraph
    y_pos = 100
    intro_text = [
        "This document provides detailed technical specifications for the XYZ-2024 model.",
        "All parameters have been tested under standard laboratory conditions.",
        "For additional information, please refer to the complete user manual."
    ]
    
    for line in intro_text:
        draw.text((50, y_pos), line, fill='black', font=font_medium)
        y_pos += 25
    
    # Table 1: Basic Specifications
    table1_y = y_pos + 20
    draw.text((50, table1_y), "Table 1: Basic Specifications", fill='black', font=font_large)
    
    table1_start = table1_y + 30
    table1_width, table1_height = 700, 200
    
    # Draw table border
    draw.rectangle([50, table1_start, 50 + table1_width, table1_start + table1_height], 
                  outline='black', width=2)
    
    # Table headers
    header_height = 35
    draw.rectangle([50, table1_start, 50 + table1_width, table1_start + header_height], 
                  fill='#E6E6E6', outline='black', width=1)
    
    # Column setup for Table 1
    col_widths = [250, 200, 250]  # Parameter, Value, Specification
    col_positions = [50, 50 + col_widths[0], 50 + col_widths[0] + col_widths[1]]
    
    # Draw column separators
    for pos in col_positions[1:]:
        draw.line([pos, table1_start, pos, table1_start + table1_height], fill='black', width=1)
    
    # Header text
    headers1 = ["Parameter", "Value", "Specification Range"]
    for i, header in enumerate(headers1):
        draw.text((col_positions[i] + 10, table1_start + 8), header, fill='black', font=font_medium)
    
    # Table 1 data
    table1_data = [
        ("Operating Temperature", "25°C", "-10°C to +60°C"),
        ("Supply Voltage", "12V DC", "10V - 15V DC"),
        ("Current Consumption", "150mA", "Max 200mA"),
        ("Frequency Range", "2.4 GHz", "2.4 - 2.485 GHz"),
        ("Output Power", "10dBm", "Max 20dBm")
    ]
    
    row_height = (table1_height - header_height) // len(table1_data)
    for i, (param, value, spec) in enumerate(table1_data):
        y = table1_start + header_height + i * row_height
        # Row separator
        if i > 0:
            draw.line([50, y, 50 + table1_width, y], fill='gray', width=1)
        
        # Data
        draw.text((col_positions[0] + 10, y + 8), param, fill='black', font=font_small)
        draw.text((col_positions[1] + 10, y + 8), value, fill='black', font=font_small)
        draw.text((col_positions[2] + 10, y + 8), spec, fill='black', font=font_small)
    
    # Table 2: Performance Metrics
    table2_y = table1_start + table1_height + 40
    draw.text((50, table2_y), "Table 2: Performance Metrics", fill='black', font=font_large)
    
    table2_start = table2_y + 30
    table2_width, table2_height = 800, 180
    
    # Draw table border
    draw.rectangle([50, table2_start, 50 + table2_width, table2_start + table2_height], 
                  outline='black', width=2)
    
    # Table 2 headers
    draw.rectangle([50, table2_start, 50 + table2_width, table2_start + header_height], 
                  fill='#E6E6E6', outline='black', width=1)
    
    # Column setup for Table 2
    col2_widths = [200, 150, 150, 150, 150]  # Metric, Min, Typical, Max, Units
    col2_positions = [50]
    for width in col2_widths[:-1]:
        col2_positions.append(col2_positions[-1] + width)
    
    # Draw column separators
    for pos in col2_positions[1:]:
        draw.line([pos, table2_start, pos, table2_start + table2_height], fill='black', width=1)
    
    # Header text
    headers2 = ["Metric", "Min", "Typical", "Max", "Units"]
    for i, header in enumerate(headers2):
        draw.text((col2_positions[i] + 10, table2_start + 8), header, fill='black', font=font_medium)
    
    # Table 2 data
    table2_data = [
        ("Sensitivity", "-85", "-82", "-80", "dBm"),
        ("Range", "50", "100", "150", "meters"),
        ("Data Rate", "1", "2", "3", "Mbps"),
        ("Latency", "1", "5", "10", "ms")
    ]
    
    row_height2 = (table2_height - header_height) // len(table2_data)
    for i, (metric, min_val, typ_val, max_val, units) in enumerate(table2_data):
        y = table2_start + header_height + i * row_height2
        # Row separator
        if i > 0:
            draw.line([50, y, 50 + table2_width, y], fill='gray', width=1)
        
        # Data
        values = [metric, min_val, typ_val, max_val, units]
        for j, value in enumerate(values):
            draw.text((col2_positions[j] + 10, y + 8), value, fill='black', font=font_small)
    
    # Footer
    footer_y = table2_start + table2_height + 30
    draw.text((50, footer_y), "Notes:", fill='black', font=font_medium)
    draw.text((50, footer_y + 20), "• All measurements taken at 25°C ambient temperature", fill='black', font=font_small)
    draw.text((50, footer_y + 35), "• Specifications subject to change without notice", fill='black', font=font_small)
    draw.text((50, footer_y + 50), "• For calibration procedures, see Section 4.2", fill='black', font=font_small)
    
    # Save test image
    test_image_path = "temp_realistic_table_test.png"
    img.save(test_image_path)
    
    try:
        print("🔍 Analyzing realistic document for table detection...")
        
        # Initialize region detector
        detector = RegionDetector(api_key)
        
        # Analyze the test image
        layout = detector.analyze_document(test_image_path, page_number=1)
        
        print(f"✅ Document analysis completed!")
        print(f"   Total regions detected: {len(layout.regions)}")
        
        # Show all detected regions
        print(f"\n📋 All Detected Regions:")
        region_counts = {}
        for i, region in enumerate(layout.regions):
            region_type = region.type.value
            region_counts[region_type] = region_counts.get(region_type, 0) + 1
            
            print(f"   {i+1}. {region.type.value}")
            print(f"      ID: {region.id}")
            print(f"      Confidence: {region.confidence:.2f}")
            print(f"      Description: {region.content_description}")
            print()
        
        # Summary by type
        print(f"📊 Region Summary:")
        for region_type, count in sorted(region_counts.items()):
            print(f"   {region_type}: {count}")
        
        # Focus on table analysis
        table_regions = [r for r in layout.regions if r.type == RegionType.TABLE]
        
        if table_regions:
            print(f"\n📊 Detailed Table Analysis:")
            print(f"   Found {len(table_regions)} table region(s)")
            
            for i, table_region in enumerate(table_regions):
                print(f"\n   📋 Table {i+1} Analysis:")
                print(f"     Region ID: {table_region.id}")
                print(f"     Detection Confidence: {table_region.confidence:.2f}")
                print(f"     Description: {table_region.content_description}")
                print(f"     Position: ({table_region.bbox.x:.3f}, {table_region.bbox.y:.3f})")
                print(f"     Size: {table_region.bbox.width:.3f} x {table_region.bbox.height:.3f}")
                
                # Detailed table structure analysis
                try:
                    print(f"     🔍 Analyzing table structure...")
                    table_structure = detector.analyze_table_structure(test_image_path, table_region)
                    
                    print(f"     ✅ Structure Analysis Results:")
                    print(f"       Dimensions: {table_structure.rows} rows × {table_structure.cols} columns")
                    print(f"       Headers detected: {table_structure.headers}")
                    print(f"       Structure confidence: {table_structure.structure_confidence:.2f}")
                    
                    if table_structure.cells:
                        print(f"       Total cells extracted: {len(table_structure.cells)}")
                        
                        # Show headers if detected
                        if table_structure.headers:
                            header_cells = [cell for cell in table_structure.cells if cell.row == 0]
                            if header_cells:
                                print(f"       Header row:")
                                for cell in sorted(header_cells, key=lambda c: c.col):
                                    print(f"         Col {cell.col}: '{cell.content}' (conf: {cell.confidence:.2f})")
                        
                        # Show some data cells
                        data_cells = [cell for cell in table_structure.cells if cell.row > 0]
                        if data_cells:
                            print(f"       Sample data cells:")
                            for cell in sorted(data_cells, key=lambda c: (c.row, c.col))[:6]:
                                print(f"         ({cell.row}, {cell.col}): '{cell.content}' (conf: {cell.confidence:.2f})")
                            
                            if len(data_cells) > 6:
                                print(f"         ... and {len(data_cells) - 6} more data cells")
                        
                        # Calculate average confidence
                        avg_confidence = sum(cell.confidence for cell in table_structure.cells) / len(table_structure.cells)
                        print(f"       Average cell confidence: {avg_confidence:.2f}")
                    
                except Exception as e:
                    print(f"     ❌ Table structure analysis failed: {e}")
                    print(f"     💡 This might happen if the table structure is too complex")
        else:
            print("ℹ️  No table regions detected")
            print("💡 Try adjusting the image content or table structure")
        
        print(f"\n🎉 Realistic table test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"🧹 Cleaned up test image: {test_image_path}")


def analyze_pdf_with_tables(pdf_path: str):
    """
    Analyze a real PDF file for table detection and extraction
    
    Args:
        pdf_path: Path to the PDF file to analyze
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Analyzing PDF: {pdf_path}")
    
    try:
        # Convert PDF to images
        print("🔄 Converting PDF to images...")
        image_paths = convert_pdf_to_images(pdf_path, dpi=300)
        
        if not image_paths:
            print("❌ Failed to convert PDF to images")
            return
        
        print(f"✅ Successfully converted {len(image_paths)} pages")
        
        # Initialize region detector
        detector = RegionDetector(api_key)
        
        total_tables = 0
        
        # Process each page
        for page_num, image_path in enumerate(image_paths, 1):
            print(f"\n{'='*50}")
            print(f"📄 Processing Page {page_num}")
            print(f"{'='*50}")
            
            try:
                # Ensure image_path is a string path, not an Image object
                if hasattr(image_path, 'save'):  # Check if it's a PIL Image object
                    # If it's an Image object, save it to a temporary file
                    temp_path = f"temp_page_{page_num}.png"
                    image_path.save(temp_path)
                    actual_image_path = temp_path
                    cleanup_temp = True
                else:
                    # It's already a file path
                    actual_image_path = str(image_path)
                    cleanup_temp = False
                
                # Verify the file exists
                if not os.path.exists(actual_image_path):
                    print(f"❌ Image file not found: {actual_image_path}")
                    continue
                
                # Analyze document layout
                layout = detector.analyze_document(actual_image_path, page_number=page_num)
                
                print(f"✅ Page {page_num} analysis completed")
                print(f"   Total regions: {len(layout.regions)}")
                
                # Count regions by type
                region_counts = {}
                for region in layout.regions:
                    region_type = region.type.value
                    region_counts[region_type] = region_counts.get(region_type, 0) + 1
                
                print(f"   Region breakdown:")
                for region_type, count in sorted(region_counts.items()):
                    print(f"     {region_type}: {count}")
                
                # Focus on tables
                table_regions = [r for r in layout.regions if r.type == RegionType.TABLE]
                
                if table_regions:
                    print(f"\n📊 Found {len(table_regions)} table(s) on page {page_num}")
                    total_tables += len(table_regions)
                    
                    for i, table_region in enumerate(table_regions):
                        print(f"\n   📋 Table {page_num}.{i+1}:")
                        print(f"     ID: {table_region.id}")
                        print(f"     Confidence: {table_region.confidence:.2f}")
                        print(f"     Description: {table_region.content_description}")
                        print(f"     Position: ({table_region.bbox.x:.3f}, {table_region.bbox.y:.3f})")
                        print(f"     Size: {table_region.bbox.width:.3f} × {table_region.bbox.height:.3f}")
                        
                        # Analyze table structure
                        try:
                            # Add size check before detailed analysis
                            estimated_cells = table_region.bbox.width * table_region.bbox.height * 100  # Rough estimate
                            if estimated_cells > 10000:  # Very large table
                                print(f"     ⚠️  Large table detected, using simplified analysis...")
                                print(f"     📊 Estimated complexity: High")
                                print(f"     💡 Consider using smaller table regions or preprocessing")
                            else:
                                print(f"     🔍 Analyzing table structure...")
                                table_structure = detector.analyze_table_structure(actual_image_path, table_region)
                                
                                print(f"     ✅ Structure Analysis Results:")
                                print(f"       Dimensions: {table_structure.rows} rows × {table_structure.cols} columns")
                                print(f"       Headers detected: {table_structure.headers}")
                                print(f"       Structure confidence: {table_structure.structure_confidence:.2f}")
                                
                                if table_structure.cells:
                                    total_cells = len(table_structure.cells)
                                    print(f"       Total cells extracted: {total_cells}")
                                    
                                    # Limit cell display for very large tables
                                    max_display_cells = 6 if total_cells > 50 else 4
                                    
                                    # Show headers if detected
                                    if table_structure.headers:
                                        header_cells = [cell for cell in table_structure.cells if cell.row == 0]
                                        if header_cells:
                                            print(f"       Header row:")
                                            for cell in sorted(header_cells, key=lambda c: c.col)[:max_display_cells]:
                                                content_preview = cell.content[:20] + "..." if len(cell.content) > 20 else cell.content
                                                print(f"         Col {cell.col}: '{content_preview}' (conf: {cell.confidence:.2f})")
                                            if len(header_cells) > max_display_cells:
                                                print(f"         ... and {len(header_cells) - max_display_cells} more header cells")
                                    
                                    # Show some data cells
                                    data_cells = [cell for cell in table_structure.cells if cell.row > 0]
                                    if data_cells:
                                        print(f"       Sample data cells:")
                                        sample_cells = sorted(data_cells, key=lambda c: (c.row, c.col))[:max_display_cells]
                                        for cell in sample_cells:
                                            content_preview = cell.content[:20] + "..." if len(cell.content) > 20 else cell.content
                                            print(f"         ({cell.row},{cell.col}): '{content_preview}' (conf: {cell.confidence:.2f})")
                                        
                                        if len(data_cells) > max_display_cells:
                                            print(f"         ... and {len(data_cells) - max_display_cells} more data cells")
                                    
                                    # Calculate average confidence
                                    avg_confidence = sum(cell.confidence for cell in table_structure.cells) / len(table_structure.cells)
                                    print(f"       Average cell confidence: {avg_confidence:.2f}")
                                    
                                    # Additional info for large tables
                                    if total_cells > 50:
                                        print(f"       💡 Large table ({total_cells} cells) - showing sample only")
                            
                        except Exception as e:
                            print(f"     ❌ Structure analysis failed: {str(e)[:200]}...")  # Truncate long error messages
                            print(f"     💡 This might be due to table complexity or size")
                            print(f"     📊 Basic table detection was successful (confidence: {table_region.confidence:.2f})")
                
                else:
                    print(f"   ℹ️  No tables detected on page {page_num}")
                
                # Clean up temporary file if created
                if cleanup_temp and os.path.exists(actual_image_path):
                    os.remove(actual_image_path)
                    
            except Exception as e:
                print(f"❌ Failed to process page {page_num}: {e}")
                # Clean up on error too
                if 'cleanup_temp' in locals() and cleanup_temp and 'actual_image_path' in locals() and os.path.exists(actual_image_path):
                    os.remove(actual_image_path)
                continue
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total pages processed: {len(image_paths)}")
        print(f"Total tables found: {total_tables}")
        print(f"Average tables per page: {total_tables / len(image_paths):.1f}")
        
    except Exception as e:
        print(f"❌ PDF analysis failed: {e}")
        import traceback
        traceback.print_exc()


def compare_document_analyzer():
    """
    Compare the standalone region detector with the full document analyzer
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Use a test PDF or create one
    pdf_path = "data/inputs/sample_Datenblatt.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"ℹ️  PDF not found: {pdf_path}")
        print("🎨 Creating test document for comparison...")
        create_realistic_table_test()
        return
    
    print("🔬 Comparing Region Detector vs Document Analyzer")
    print("=" * 60)
    
    try:
        # Method 1: Region Detector only
        print("\n🔍 Method 1: Using Region Detector")
        print("-" * 40)
        
        import time
        start_time = time.time()
        
        # Convert PDF and analyze with region detector
        image_paths = convert_pdf_to_images(pdf_path, dpi=300)
        detector = RegionDetector(api_key)
        
        method1_results = []
        for page_num, image_path in enumerate(image_paths, 1):
            layout = detector.analyze_document(image_path, page_number=page_num)
            table_regions = [r for r in layout.regions if r.type == RegionType.TABLE]
            method1_results.append({
                'page': page_num,
                'total_regions': len(layout.regions),
                'tables': len(table_regions)
            })
        
        method1_time = time.time() - start_time
        
        print(f"✅ Region Detector completed in {method1_time:.2f}s")
        for result in method1_results:
            print(f"   Page {result['page']}: {result['total_regions']} regions, {result['tables']} tables")
        
        # Method 2: Full Document Analyzer
        print("\n📄 Method 2: Using Document Analyzer")
        print("-" * 40)
        
        start_time = time.time()
        
        analyzer = DocumentAnalyzer(api_key)
        output_dir = "temp_comparison_output"
        
        results = analyzer.analyze_document(
            input_path=pdf_path,
            output_dir=output_dir,
            extract_tables=True,
            create_visualizations=True
        )
        
        method2_time = time.time() - start_time
        
        print(f"✅ Document Analyzer completed in {method2_time:.2f}s")
        print(f"   Total pages: {results['document_info']['total_pages']}")
        print(f"   Total regions: {results['analysis_summary']['total_regions_detected']}")
        print(f"   Total tables: {results['analysis_summary']['total_table_regions']}")
        print(f"   Successful table analyses: {results['analysis_summary']['successful_table_analyses']}")
        
        # Comparison summary
        print(f"\n⚖️  COMPARISON SUMMARY")
        print("-" * 40)
        print(f"Region Detector time: {method1_time:.2f}s")
        print(f"Document Analyzer time: {method2_time:.2f}s")
        print(f"Speed difference: {method2_time/method1_time:.1f}x")
        print()
        print("Region Detector pros:")
        print("  ✅ Faster execution")
        print("  ✅ Lower memory usage")
        print("  ✅ More flexible for custom workflows")
        print()
        print("Document Analyzer pros:")
        print("  ✅ Complete pipeline with visualizations")
        print("  ✅ Structured output files")
        print("  ✅ Built-in error handling and recovery")
        print("  ✅ Comprehensive analysis summaries")
        
        # Clean up
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"🧹 Cleaned up: {output_dir}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function with example menu
    """
    
    print("🚀 PDF2Ontology Table Analysis Examples")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    while True:
        print("\n📋 Available Examples:")
        print("1. 🎨 Realistic Table Test (No external files needed)")
        print("2. 📄 Analyze PDF with Tables")
        print("3. 🔬 Compare Region Detector vs Document Analyzer") 
        print("4. ❌ Exit")
        
        try:
            choice = input("\nSelect an example (1-4): ").strip()
            
            if choice == "1":
                create_realistic_table_test()
            elif choice == "2":
                pdf_path = input("Enter PDF path (or press Enter for default): ").strip()
                if not pdf_path:
                    pdf_path = "data/inputs/sample_Datenblatt.pdf"
                analyze_pdf_with_tables(pdf_path)
            elif choice == "3":
                compare_document_analyzer()
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()