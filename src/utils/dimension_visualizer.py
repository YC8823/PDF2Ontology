# src/utils/dimension_visualizer.py

import json
import cv2
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DimensionVisualizer:
    """
    可视化工具：在工程图上绘制尺寸标注和锚点
    """
    
    # 不同方向的尺寸线使用不同的颜色（BGR格式）
    ORIENTATION_COLORS = {
        'horizontal': (0, 255, 0),      # 绿色
        'vertical': (255, 0, 0),        # 蓝色
        'diameter': (0, 165, 255),      # 橙色
        'other': (255, 0, 255)          # 紫色
    }
    
    def __init__(self, 
                 line_thickness: int = 2,
                 anchor_radius: int = 5,
                 font_scale: float = 0.5,
                 font_thickness: int = 1):
        """
        初始化可视化器
        
        Args:
            line_thickness: 尺寸线的粗细
            anchor_radius: 锚点圆圈的半径
            font_scale: 文字大小比例
            font_thickness: 文字粗细
        """
        self.line_thickness = line_thickness
        self.anchor_radius = anchor_radius
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_dimension_line(self, 
                          image: np.ndarray, 
                          start: Tuple[int, int], 
                          end: Tuple[int, int],
                          color: Tuple[int, int, int]) -> np.ndarray:
        """
        绘制尺寸线
        
        Args:
            image: 输入图像
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            color: BGR颜色
            
        Returns:
            绘制后的图像
        """
        cv2.line(image, start, end, color, self.line_thickness)
        return image
    
    def draw_anchor_point(self,
                         image: np.ndarray,
                         point: Tuple[int, int],
                         color: Tuple[int, int, int]) -> np.ndarray:
        """
        绘制锚点（用实心圆表示）
        
        Args:
            image: 输入图像
            point: 锚点坐标 (x, y)
            color: BGR颜色
            
        Returns:
            绘制后的图像
        """
        cv2.circle(image, point, self.anchor_radius, color, -1)  # -1表示填充
        # 绘制白色边框以提高可见度
        cv2.circle(image, point, self.anchor_radius + 1, (255, 255, 255), 1)
        return image
    
    def draw_label(self,
                  image: np.ndarray,
                  text: str,
                  position: Tuple[int, int],
                  color: Tuple[int, int, int],
                  bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        在指定位置绘制文字标签（带背景）
        
        Args:
            image: 输入图像
            text: 要显示的文字
            position: 文字位置 (x, y)
            color: 文字颜色
            bg_color: 背景颜色（可选）
            
        Returns:
            绘制后的图像
        """
        # 获取文字大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        
        # 如果提供了背景颜色，绘制背景矩形
        if bg_color:
            padding = 2
            cv2.rectangle(
                image,
                (position[0] - padding, position[1] - text_height - padding),
                (position[0] + text_width + padding, position[1] + baseline + padding),
                bg_color,
                -1
            )
        
        # 绘制文字
        cv2.putText(
            image, text, position, self.font,
            self.font_scale, color, self.font_thickness, cv2.LINE_AA
        )
        
        return image
    
    def get_label_position(self,
                          start: Tuple[int, int],
                          end: Tuple[int, int],
                          orientation: str,
                          offset: int = 15) -> Tuple[int, int]:
        """
        计算标签的最佳位置（在尺寸线中点附近）
        
        Args:
            start: 起点坐标
            end: 终点坐标
            orientation: 尺寸方向
            offset: 标签偏移距离
            
        Returns:
            标签位置 (x, y)
        """
        # 计算中点
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2
        
        # 根据方向调整标签位置
        if orientation == 'horizontal':
            # 水平线的标签放在上方
            return (mid_x, mid_y - offset)
        elif orientation == 'vertical':
            # 垂直线的标签放在右侧
            return (mid_x + offset, mid_y)
        else:
            # 其他方向的标签放在右上方
            return (mid_x + offset, mid_y - offset)
    
    def visualize_single_dimension(self,
                                   image: np.ndarray,
                                   dimension: Dict) -> np.ndarray:
        """
        可视化单个尺寸标注
        
        Args:
            image: 输入图像
            dimension: 尺寸信息字典
            
        Returns:
            绘制后的图像
        """
        # 提取信息
        param_name = dimension.get('parameter_name', 'Unknown')
        value = dimension.get('value')
        unit = dimension.get('unit', '')
        orientation = dimension.get('orientation', 'other')
        
        start_anchor = dimension.get('start_anchor', {})
        end_anchor = dimension.get('end_anchor', {})
        
        start_point = (start_anchor.get('x', 0), start_anchor.get('y', 0))
        end_point = (end_anchor.get('x', 0), end_anchor.get('y', 0))
        
        # 获取颜色
        color = self.ORIENTATION_COLORS.get(orientation, self.ORIENTATION_COLORS['other'])
        
        # 绘制尺寸线
        image = self.draw_dimension_line(image, start_point, end_point, color)
        
        # 绘制锚点
        image = self.draw_anchor_point(image, start_point, color)
        image = self.draw_anchor_point(image, end_point, color)
        
        # 准备标签文字
        if value is not None and unit:
            label_text = f"{param_name}={value}{unit}"
        elif value is not None:
            label_text = f"{param_name}={value}"
        else:
            label_text = param_name
        
        # 计算标签位置并绘制
        label_pos = self.get_label_position(start_point, end_point, orientation)
        image = self.draw_label(image, label_text, label_pos, color, bg_color=(0, 0, 0))
        
        return image
    
    def visualize_image_dimensions(self,
                                   image_path: str,
                                   dimensions: List[Dict],
                                   output_path: str) -> bool:
        """
        可视化一张图像的所有尺寸标注
        
        Args:
            image_path: 输入图像路径
            dimensions: 尺寸列表
            output_path: 输出图像路径
            
        Returns:
            是否成功
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return False
            
            logger.info(f"处理图像: {image_path}")
            logger.info(f"  图像尺寸: {image.shape[1]}x{image.shape[0]}")
            logger.info(f"  尺寸标注数量: {len(dimensions)}")
            
            # 绘制所有尺寸标注
            for dim in dimensions:
                image = self.visualize_single_dimension(image, dim)
            
            # 添加图例
            image = self.add_legend(image)
            
            # 保存结果
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            logger.info(f"  ✓ 保存到: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理图像时出错 {image_path}: {e}")
            return False
    
    def add_legend(self, image: np.ndarray, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        在图像上添加图例
        
        Args:
            image: 输入图像
            position: 图例起始位置
            
        Returns:
            添加图例后的图像
        """
        x, y = position
        line_height = 25
        
        legend_items = [
            ('Horizontal', 'horizontal'),
            ('Vertical', 'vertical'),
            ('Diameter', 'diameter'),
            ('Other', 'other')
        ]
        
        for i, (label, orientation) in enumerate(legend_items):
            color = self.ORIENTATION_COLORS[orientation]
            current_y = y + i * line_height
            
            # 绘制颜色示例线
            cv2.line(image, (x, current_y), (x + 30, current_y), color, self.line_thickness)
            
            # 绘制文字
            self.draw_label(
                image, 
                label, 
                (x + 40, current_y + 5),
                (255, 255, 255),
                bg_color=(0, 0, 0)
            )
        
        return image
    
    def visualize_from_json(self,
                           json_path: str,
                           output_dir: str,
                           base_image_dir: str = "data/intermediate_results/t58740/cropped_images"):
        """
        从visual_analysis_results.json批量生成可视化结果
        
        Args:
            json_path: visual_analysis_results.json文件路径
            output_dir: 输出目录
            base_image_dir: 裁剪图像的基础目录
        """
        logger.info(f"开始批量可视化...")
        logger.info(f"  JSON文件: {json_path}")
        logger.info(f"  输出目录: {output_dir}")
        
        # 加载JSON数据
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"  加载了 {len(results)} 条记录")
        except Exception as e:
            logger.error(f"无法加载JSON文件: {e}")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        success_count = 0
        skip_count = 0
        error_count = 0
        
        # 遍历每条记录
        for record in results:
            image_id = record.get('image_id', 'unknown')
            
            # 检查是否成功提取了尺寸
            if record.get('visual_analysis_status') != 'success':
                logger.debug(f"跳过 {image_id}: 分析状态不是success")
                skip_count += 1
                continue
            
            all_dimensions = record.get('all_device_dimensions', [])
            if not all_dimensions or len(all_dimensions) == 0:
                logger.debug(f"跳过 {image_id}: 没有尺寸数据")
                skip_count += 1
                continue
            
            # 获取图像路径
            image_path = record.get('cropped_image_path', '')
            
            # 处理路径（Windows路径可能需要转换）
            if '\\' in image_path:
                image_path = image_path.replace('\\', '/')
            
            # 如果路径不存在，尝试使用base_image_dir
            if not os.path.exists(image_path):
                # 提取文件名
                image_filename = os.path.basename(image_path)
                image_path = os.path.join(base_image_dir, image_filename)
            
            if not os.path.exists(image_path):
                logger.warning(f"图像文件不存在: {image_path}")
                error_count += 1
                continue
            
            # 提取所有尺寸
            all_dims = []
            for device_dim in all_dimensions:
                dims = device_dim.get('dimensions', [])
                all_dims.extend(dims)
            
            if len(all_dims) == 0:
                logger.debug(f"跳过 {image_id}: dimensions列表为空")
                skip_count += 1
                continue
            
            # 生成输出路径
            output_filename = f"{image_id}_annotated.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 可视化
            if self.visualize_image_dimensions(image_path, all_dims, output_path):
                success_count += 1
            else:
                error_count += 1
        
        # 输出统计信息
        logger.info(f"\n{'='*60}")
        logger.info(f"可视化完成!")
        logger.info(f"  成功: {success_count}")
        logger.info(f"  跳过: {skip_count}")
        logger.info(f"  错误: {error_count}")
        logger.info(f"  总计: {len(results)}")
        logger.info(f"{'='*60}")


def main():
    """
    主函数：命令行执行入口
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="可视化尺寸标注和锚点"
    )
    parser.add_argument(
        '--input_json',
        type=str,
        default='./data/intermediate_results/t58740/visual_analysis_results.json',
        help='visual_analysis_results.json文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/intermediate_results/t58740/visualized_dimensions',
        help='可视化结果输出目录'
    )
    parser.add_argument(
        '--base_image_dir',
        type=str,
        default='data/intermediate_results/t58740/cropped_images',
        help='裁剪图像的基础目录'
    )
    parser.add_argument(
        '--line_thickness',
        type=int,
        default=2,
        help='尺寸线粗细'
    )
    parser.add_argument(
        '--anchor_radius',
        type=int,
        default=5,
        help='锚点半径'
    )
    parser.add_argument(
        '--font_scale',
        type=float,
        default=0.5,
        help='文字大小比例'
    )
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = DimensionVisualizer(
        line_thickness=args.line_thickness,
        anchor_radius=args.anchor_radius,
        font_scale=args.font_scale
    )
    
    # 执行批量可视化
    visualizer.visualize_from_json(
        json_path=args.input_json,
        output_dir=args.output_dir,
        base_image_dir=args.base_image_dir
    )


if __name__ == "__main__":
    main()
