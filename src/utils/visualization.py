"""
Visualization utilities for annotating detected regions
"""

import os, sys
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import colorsys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from ..core.models import DocumentRegion, RegionType

class RegionVisualizer:
    """Creates visual annotations for detected document regions"""
    
    # Color scheme for different region types
    TYPE_COLORS = {
        RegionType.TEXT: "#2E86C1",      # Blue
        RegionType.TABLE: "#E74C3C",     # Red  
        RegionType.IMAGE: "#27AE60",     # Green
        RegionType.HEADER: "#8E44AD",    # Purple
        RegionType.FOOTER: "#F39C12",    # Orange
        RegionType.TITLE: "#C0392B",     # Dark Red
        RegionType.CAPTION: "#16A085",   # Teal
        RegionType.SIDEBAR: "#7D3C98",   # Dark Purple
    }
    
    def __init__(self, line_width: int = 3, font_size: int = 16, 
                 show_confidence: bool = True, show_ids: bool = True):
        self.line_width = line_width
        self.font_size = font_size
        self.show_confidence = show_confidence
        self.show_ids = show_ids
        
        # Try to load font
        try:
            self.font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                self.font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                self.font = ImageFont.load_default()
    
    def annotate_regions(self, image_path: str, regions: List[DocumentRegion], 
                        output_path: str, style: str = "detailed") -> str:
        """Annotate image with detected regions"""
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        if style == "minimal":
            self._draw_minimal_annotations(draw, regions)
        elif style == "detailed":
            self._draw_detailed_annotations(draw, regions)
        elif style == "confidence_heatmap":
            self._draw_confidence_heatmap(draw, regions)
        else:
            raise ValueError(f"Unknown annotation style: {style}")
        
        # Save annotated image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, 'PNG', quality=95)
        
        return output_path
    
    def _draw_minimal_annotations(self, draw: ImageDraw.Draw, regions: List[DocumentRegion]):
        """Draw minimal bounding boxes only"""
        
        for region in regions:
            bbox = region.bbox
            color = self.TYPE_COLORS.get(region.region_type, "#000000")
            
            # Draw bounding box
            draw.rectangle(
                [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height],
                outline=color,
                width=self.line_width
            )
    
    def _draw_detailed_annotations(self, draw: ImageDraw.Draw, regions: List[DocumentRegion]):
        """Draw detailed annotations with labels and metadata"""
        
        for i, region in enumerate(regions):
            bbox = region.bbox
            color = self.TYPE_COLORS.get(region.region_type, "#000000")
            
            # Draw bounding box
            draw.rectangle(
                [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height],
                outline=color,
                width=self.line_width
            )
            
            # Create label text
            label_parts = []
            if self.show_ids:
                label_parts.append(f"{region.region_type.value}_{i}")
            if self.show_confidence:
                label_parts.append(f"{region.confidence:.2f}")
            
            label_text = " | ".join(label_parts)
            
            # Draw label background
            text_bbox = draw.textbbox((0, 0), label_text, font=self.font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            label_x = bbox.x
            label_y = max(0, bbox.y - text_height - 5)
            
            # Background rectangle
            draw.rectangle(
                [label_x, label_y, label_x + text_width + 8, label_y + text_height + 4],
                fill=color,
                outline=color
            )
            
            # Label text
            draw.text(
                (label_x + 4, label_y + 2),
                label_text,
                fill="white",
                font=self.font
            )
            
            # Draw reading order if available
            if region.reading_order is not None:
                order_text = str(region.reading_order)
                
                # Draw reading order circle
                center_x, center_y = bbox.center()
                circle_radius = 15
                
                draw.ellipse(
                    [center_x - circle_radius, center_y - circle_radius,
                     center_x + circle_radius, center_y + circle_radius],
                    fill="white",
                    outline=color,
                    width=2
                )
                
                # Center the text in circle
                order_bbox = draw.textbbox((0, 0), order_text, font=self.font)
                order_width = order_bbox[2] - order_bbox[0]
                order_height = order_bbox[3] - order_bbox[1]
                
                draw.text(
                    (center_x - order_width // 2, center_y - order_height // 2),
                    order_text,
                    fill=color,
                    font=self.font
                )
    
    def _draw_confidence_heatmap(self, draw: ImageDraw.Draw, regions: List[DocumentRegion]):
        """Draw regions with confidence-based color intensity"""
        
        for region in regions:
            bbox = region.bbox
            base_color = self.TYPE_COLORS.get(region.region_type, "#000000")
            
            # Convert hex to RGB
            base_rgb = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
            
            # Adjust opacity based on confidence
            alpha = int(255 * region.confidence * 0.3)  # 30% max opacity
            
            # Draw filled rectangle with transparency effect
            overlay = Image.new('RGBA', (bbox.width, bbox.height), base_rgb + (alpha,))
            
            # This is a simplified approach - for true alpha blending, you'd need more complex operations
            draw.rectangle(
                [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height],
                outline=base_color,
                width=max(1, int(self.line_width * region.confidence))
            )
    
    def create_region_summary(self, regions: List[DocumentRegion], output_path: str):
        """Create a summary visualization showing region statistics"""
        
        # Count regions by type
        type_counts = {}
        confidence_stats = {}
        
        for region in regions:
            region_type = region.region_type.value
            type_counts[region_type] = type_counts.get(region_type, 0) + 1
            
            if region_type not in confidence_stats:
                confidence_stats[region_type] = []
            confidence_stats[region_type].append(region.confidence)
        
        # Create summary image (simple text-based for now)
        summary_height = len(type_counts) * 30 + 100
        summary_image = Image.new('RGB', (400, summary_height), 'white')
        draw = ImageDraw.Draw(summary_image)
        
        y_offset = 20
        draw.text((20, y_offset), "Region Detection Summary", fill="black", font=self.font)
        y_offset += 40
        
        for region_type, count in type_counts.items():
            avg_confidence = sum(confidence_stats[region_type]) / len(confidence_stats[region_type])
            color = self.TYPE_COLORS.get(RegionType(region_type), "#000000")
            
            # Draw color indicator
            draw.rectangle([20, y_offset, 35, y_offset + 15], fill=color)
            
            # Draw statistics
            text = f"{region_type}: {count} regions (avg conf: {avg_confidence:.2f})"
            draw.text((45, y_offset), text, fill="black", font=self.font)
            y_offset += 25
        
        summary_image.save(output_path, 'PNG')
        return output_path