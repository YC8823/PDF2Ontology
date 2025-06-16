"""
Visualization utilities for annotating detected regions
Updated for new Pydantic models
"""

import os
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import colorsys

# Import new Pydantic models
from ..pydantic_models.region_models import DocumentRegion
from ..pydantic_models.enums import RegionType

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
        RegionType.OTHER: "#95A5A6",     # Gray
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
                try:
                    # Linux font paths
                    self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    self.font = ImageFont.load_default()
    
    def annotate_regions(self, image_path: str, regions: List[DocumentRegion], 
                        output_path: str, style: str = "detailed") -> str:
        """Annotate image with detected regions"""
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Get image dimensions for coordinate conversion
        img_width, img_height = image.size
        
        if style == "minimal":
            self._draw_minimal_annotations(draw, regions, img_width, img_height)
        elif style == "detailed":
            self._draw_detailed_annotations(draw, regions, img_width, img_height)
        elif style == "confidence_heatmap":
            self._draw_confidence_heatmap(draw, regions, img_width, img_height)
        else:
            raise ValueError(f"Unknown annotation style: {style}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, 'PNG', quality=95)
        
        return output_path
    
    def _convert_relative_to_absolute(self, region: DocumentRegion, img_width: int, img_height: int) -> tuple:
        """Convert relative coordinates (0.0-1.0) to absolute pixel coordinates"""
        
        bbox = region.bbox
        
        # Convert relative to absolute coordinates
        abs_x = int(bbox.x * img_width)
        abs_y = int(bbox.y * img_height)
        abs_width = int(bbox.width * img_width)
        abs_height = int(bbox.height * img_height)
        
        # Ensure coordinates are within image bounds
        abs_x = max(0, min(abs_x, img_width - 1))
        abs_y = max(0, min(abs_y, img_height - 1))
        abs_width = max(1, min(abs_width, img_width - abs_x))
        abs_height = max(1, min(abs_height, img_height - abs_y))
        
        return abs_x, abs_y, abs_width, abs_height
    
    def _draw_minimal_annotations(self, draw: ImageDraw.Draw, regions: List[DocumentRegion], 
                                img_width: int, img_height: int):
        """Draw minimal bounding boxes only"""
        
        for region in regions:
            abs_x, abs_y, abs_width, abs_height = self._convert_relative_to_absolute(
                region, img_width, img_height
            )
            
            # Use 'type' attribute instead of 'region_type'
            color = self.TYPE_COLORS.get(region.type, "#000000")
            
            # Draw bounding box
            draw.rectangle(
                [abs_x, abs_y, abs_x + abs_width, abs_y + abs_height],
                outline=color,
                width=self.line_width
            )
    
    def _draw_detailed_annotations(self, draw: ImageDraw.Draw, regions: List[DocumentRegion],
                                 img_width: int, img_height: int):
        """Draw detailed annotations with labels and metadata"""
        
        for i, region in enumerate(regions):
            abs_x, abs_y, abs_width, abs_height = self._convert_relative_to_absolute(
                region, img_width, img_height
            )
            
            # Use 'type' attribute instead of 'region_type'
            color = self.TYPE_COLORS.get(region.type, "#000000")
            
            # Draw bounding box
            draw.rectangle(
                [abs_x, abs_y, abs_x + abs_width, abs_y + abs_height],
                outline=color,
                width=self.line_width
            )
            
            # Create label text
            label_parts = []
            if self.show_ids:
                label_parts.append(f"{region.type.value}_{i}")
            if self.show_confidence:
                label_parts.append(f"{region.confidence:.2f}")
            
            label_text = " | ".join(label_parts)
            
            # Draw label background
            try:
                text_bbox = draw.textbbox((0, 0), label_text, font=self.font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(label_text, font=self.font)
            
            label_x = abs_x
            label_y = max(0, abs_y - text_height - 5)
            
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
            if hasattr(region, 'reading_order') and region.reading_order is not None:
                order_text = str(region.reading_order)
                
                # Draw reading order circle at region center
                center_x = abs_x + abs_width // 2
                center_y = abs_y + abs_height // 2
                circle_radius = 15
                
                draw.ellipse(
                    [center_x - circle_radius, center_y - circle_radius,
                     center_x + circle_radius, center_y + circle_radius],
                    fill="white",
                    outline=color,
                    width=2
                )
                
                # Center the text in circle
                try:
                    order_bbox = draw.textbbox((0, 0), order_text, font=self.font)
                    order_width = order_bbox[2] - order_bbox[0]
                    order_height = order_bbox[3] - order_bbox[1]
                except:
                    order_width, order_height = draw.textsize(order_text, font=self.font)
                
                draw.text(
                    (center_x - order_width // 2, center_y - order_height // 2),
                    order_text,
                    fill=color,
                    font=self.font
                )
    
    def _draw_confidence_heatmap(self, draw: ImageDraw.Draw, regions: List[DocumentRegion],
                               img_width: int, img_height: int):
        """Draw regions with confidence-based color intensity"""
        
        for region in regions:
            abs_x, abs_y, abs_width, abs_height = self._convert_relative_to_absolute(
                region, img_width, img_height
            )
            
            # Use 'type' attribute instead of 'region_type'
            base_color = self.TYPE_COLORS.get(region.type, "#000000")
            
            # Convert hex to RGB
            base_rgb = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
            
            # Adjust line width based on confidence
            confidence_line_width = max(1, int(self.line_width * region.confidence))
            
            # Draw rectangle with confidence-based line width
            draw.rectangle(
                [abs_x, abs_y, abs_x + abs_width, abs_y + abs_height],
                outline=base_color,
                width=confidence_line_width
            )
            
            # Optional: Add semi-transparent fill based on confidence
            if region.confidence > 0.7:  # Only for high-confidence regions
                # Create a lighter version of the color for fill
                fill_color = f"#{base_rgb[0]:02x}{base_rgb[1]:02x}{base_rgb[2]:02x}"
                
                # Draw with reduced opacity effect (simplified)
                inner_margin = 2
                draw.rectangle(
                    [abs_x + inner_margin, abs_y + inner_margin, 
                     abs_x + abs_width - inner_margin, abs_y + abs_height - inner_margin],
                    outline=None,
                    fill=None,  # We can't easily do alpha blending with PIL alone
                    width=0
                )
    
    def create_region_summary(self, regions: List[DocumentRegion], output_path: str):
        """Create a summary visualization showing region statistics"""
        
        # Count regions by type
        type_counts = {}
        confidence_stats = {}
        
        for region in regions:
            # Use 'type' attribute instead of 'region_type'
            region_type = region.type.value
            type_counts[region_type] = type_counts.get(region_type, 0) + 1
            
            if region_type not in confidence_stats:
                confidence_stats[region_type] = []
            confidence_stats[region_type].append(region.confidence)
        
        # Create summary image (simple text-based for now)
        summary_height = max(300, len(type_counts) * 30 + 100)
        summary_image = Image.new('RGB', (500, summary_height), 'white')
        draw = ImageDraw.Draw(summary_image)
        
        # Title
        y_offset = 20
        title_text = "Region Detection Summary"
        draw.text((20, y_offset), title_text, fill="black", font=self.font)
        y_offset += 40
        
        # Total regions
        total_regions = len(regions)
        total_text = f"Total regions detected: {total_regions}"
        draw.text((20, y_offset), total_text, fill="black", font=self.font)
        y_offset += 30
        
        # Region type breakdown
        for region_type, count in sorted(type_counts.items()):
            if confidence_stats[region_type]:  # Check if list is not empty
                avg_confidence = sum(confidence_stats[region_type]) / len(confidence_stats[region_type])
            else:
                avg_confidence = 0.0
                
            # Get color for this region type
            try:
                region_enum = RegionType(region_type)
                color = self.TYPE_COLORS.get(region_enum, "#000000")
            except ValueError:
                color = "#000000"
            
            # Draw color indicator rectangle
            draw.rectangle([20, y_offset, 35, y_offset + 15], fill=color)
            
            # Draw statistics text
            text = f"{region_type}: {count} regions (avg conf: {avg_confidence:.2f})"
            draw.text((45, y_offset), text, fill="black", font=self.font)
            y_offset += 25
        
        # Add overall statistics
        y_offset += 10
        if regions:
            overall_avg_conf = sum(r.confidence for r in regions) / len(regions)
            overall_text = f"Overall average confidence: {overall_avg_conf:.2f}"
            draw.text((20, y_offset), overall_text, fill="blue", font=self.font)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_image.save(output_path, 'PNG')
        return output_path
    
    def create_confidence_chart(self, regions: List[DocumentRegion], output_path: str):
        """Create a simple confidence distribution chart"""
        
        if not regions:
            return output_path
        
        # Create confidence histogram
        confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_counts = [0] * (len(confidence_bins) - 1)
        
        for region in regions:
            for i in range(len(confidence_bins) - 1):
                if confidence_bins[i] <= region.confidence < confidence_bins[i + 1]:
                    bin_counts[i] += 1
                    break
            else:
                # Handle confidence = 1.0 case
                if region.confidence == 1.0:
                    bin_counts[-1] += 1
        
        # Create chart image
        chart_width, chart_height = 400, 300
        chart_image = Image.new('RGB', (chart_width, chart_height), 'white')
        draw = ImageDraw.Draw(chart_image)
        
        # Chart title
        draw.text((10, 10), "Confidence Distribution", fill="black", font=self.font)
        
        # Draw bars
        bar_width = (chart_width - 60) // len(bin_counts)
        max_count = max(bin_counts) if bin_counts else 1
        
        for i, count in enumerate(bin_counts):
            x = 30 + i * bar_width
            bar_height = int((count / max_count) * (chart_height - 80)) if max_count > 0 else 0
            y = chart_height - 30 - bar_height
            
            # Draw bar
            draw.rectangle([x, y, x + bar_width - 5, chart_height - 30], 
                         fill="#3498db", outline="#2980b9")
            
            # Draw count label
            if count > 0:
                count_text = str(count)
                try:
                    text_bbox = draw.textbbox((0, 0), count_text, font=self.font)
                    text_width = text_bbox[2] - text_bbox[0]
                except:
                    text_width, _ = draw.textsize(count_text, font=self.font)
                
                draw.text((x + (bar_width - text_width) // 2, y - 20), 
                         count_text, fill="black", font=self.font)
            
            # Draw bin label
            bin_label = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
            try:
                label_bbox = draw.textbbox((0, 0), bin_label, font=self.font)
                label_width = label_bbox[2] - label_bbox[0]
            except:
                label_width, _ = draw.textsize(bin_label, font=self.font)
            
            draw.text((x + (bar_width - label_width) // 2, chart_height - 25), 
                     bin_label, fill="black", font=self.font)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        chart_image.save(output_path, 'PNG')
        return output_path