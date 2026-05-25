# sparql_gui.py
"""
Simple SPARQL Query GUI for TTL Ontology
支持选择URI导出格式
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
import pandas as pd
from rdflib import Graph, Namespace, URIRef
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SPARQLQueryGUI:
    """SPARQL查询GUI界面"""
    
    def __init__(self, ttl_path: str, output_dir: str):
        """
        初始化GUI
        
        Args:
            ttl_path: TTL文件路径
            output_dir: 输出目录
        """
        self.ttl_path = ttl_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前查询结果（两个版本）
        self.current_result_full = None      # 完整URI版本
        self.current_result_shortened = None  # 缩写版本
        
        # 加载ontology
        self.load_ontology()
        
        # 创建GUI
        self.create_gui()
    
    def load_ontology(self):
        """加载ontology"""
        logger.info(f"Loading ontology from: {self.ttl_path}")
        self.graph = Graph()
        
        try:
            self.graph.parse(self.ttl_path, format='turtle')
            
            # 定义命名空间
            self.ns = Namespace("http://www.semanticweb.org/yanha/ontologies/2025/7/untitled-ontology-26#")
            self.graph.bind("", self.ns)
            
            # 提取所有命名空间前缀（用于URI缩写）
            self.namespaces = dict(self.graph.namespaces())
            logger.info(f"Loaded namespaces: {list(self.namespaces.keys())}")
            
            logger.info(f"✓ Loaded {len(self.graph)} triples")
            self.ontology_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            self.ontology_loaded = False
            messagebox.showerror("Error", f"Failed to load ontology:\n{e}")
    
    def shorten_uri(self, value):
        """
        将完整URI缩写为前缀形式
        
        Args:
            value: 可能是URI、Literal或其他类型
            
        Returns:
            缩写后的字符串
        """
        if not isinstance(value, URIRef):
            return str(value) if value is not None else ""
        
        uri_str = str(value)
        
        # 尝试使用已知的命名空间前缀
        for prefix, namespace in self.namespaces.items():
            namespace_str = str(namespace)
            if uri_str.startswith(namespace_str):
                local_name = uri_str[len(namespace_str):]
                if prefix:  # 有前缀名
                    return f"{prefix}:{local_name}"
                else:  # 默认命名空间（空前缀）
                    return f":{local_name}"
        
        # 如果没有匹配的前缀，尝试从URI中分离本地名
        if '#' in uri_str:
            base, local = uri_str.rsplit('#', 1)
            return f"<...#{local}>"
        elif '/' in uri_str:
            base, local = uri_str.rsplit('/', 1)
            return f"<.../{local}>"
        
        # 如果都不匹配，返回完整URI（用尖括号包裹）
        return f"<{uri_str}>"
    
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("SPARQL Query Interface")
        self.root.geometry("1200x850")
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # ==================== 顶部信息栏 ====================
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(fill=tk.X)
        
        ttk.Label(
            info_frame, 
            text=f"Ontology: {Path(self.ttl_path).name}",
            font=("Arial", 10, "bold")
        ).pack(anchor=tk.W)
        
        ttk.Label(
            info_frame,
            text=f"Triples: {len(self.graph) if self.ontology_loaded else 'N/A'}  |  Output: {self.output_dir}",
            font=("Arial", 9),
            foreground="gray"
        ).pack(anchor=tk.W)
        
        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, padx=10)
        
        # ==================== 查询输入区域 ====================
        query_frame = ttk.LabelFrame(self.root, text="SPARQL Query", padding="10")
        query_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(5, 5))
        
        # 查询文本框
        self.query_text = scrolledtext.ScrolledText(
            query_frame,
            height=10,
            font=("Courier New", 10),
            wrap=tk.WORD
        )
        self.query_text.pack(fill=tk.BOTH, expand=True)
        
        # 插入默认查询模板
        default_query = """PREFIX : <http://www.semanticweb.org/yanha/ontologies/2025/7/untitled-ontology-26#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subject ?predicate ?object
WHERE {
    ?subject ?predicate ?object .
}
LIMIT 100"""
        
        self.query_text.insert("1.0", default_query)
        
        # 按钮栏
        button_frame = ttk.Frame(query_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 左侧按钮
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)
        
        ttk.Button(
            left_buttons,
            text="▶ Execute Query",
            command=self.execute_query,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            left_buttons,
            text="Clear",
            command=self.clear_query,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            left_buttons,
            text="Load from File...",
            command=self.load_query_from_file,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # 状态标签（右侧）
        self.status_label = ttk.Label(
            button_frame,
            text="Ready",
            foreground="green",
            font=("Arial", 9, "bold")
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # ==================== 结果显示区域 ====================
        result_frame = ttk.LabelFrame(self.root, text="Query Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 结果信息栏和导出按钮在同一行（置顶）
        top_result_bar = ttk.Frame(result_frame)
        top_result_bar.pack(fill=tk.X, pady=(0, 10))
        
        # 左侧：结果信息
        self.result_info_label = ttk.Label(
            top_result_bar,
            text="No results yet",
            font=("Arial", 9)
        )
        self.result_info_label.pack(side=tk.LEFT)
        
        # 右侧：导出按钮组
        export_buttons = ttk.Frame(top_result_bar)
        export_buttons.pack(side=tk.RIGHT)
        
        ttk.Label(export_buttons, text="Export as:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            export_buttons,
            text="📄 CSV",
            command=lambda: self.export_results('csv'),
            width=8
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            export_buttons,
            text="📊 Excel",
            command=lambda: self.export_results('excel'),
            width=8
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            export_buttons,
            text="{ } JSON",
            command=lambda: self.export_results('json'),
            width=8
        ).pack(side=tk.LEFT, padx=2)
        
        # 文件名和导出选项行
        filename_bar = ttk.Frame(result_frame)
        filename_bar.pack(fill=tk.X, pady=(0, 10))
        
        # 左侧：文件名
        ttk.Label(filename_bar, text="Filename:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.filename_entry = ttk.Entry(filename_bar, width=40)
        self.filename_entry.pack(side=tk.LEFT, padx=5)
        self.filename_entry.insert(0, self.generate_default_filename())
        
        ttk.Label(
            filename_bar, 
            text="(extension will be added automatically)", 
            font=("Arial", 8), 
            foreground="gray"
        ).pack(side=tk.LEFT, padx=5)
        
        # 右侧：URI格式选择
        ttk.Label(filename_bar, text="|", foreground="gray").pack(side=tk.LEFT, padx=10)
        
        self.export_shortened_var = tk.BooleanVar(value=True)  # 默认导出缩写版本
        
        ttk.Checkbutton(
            filename_bar,
            text="Export with shortened URIs (e.g., :Device, rdf:type)",
            variable=self.export_shortened_var
        ).pack(side=tk.LEFT, padx=5)
        
        # 分隔线
        ttk.Separator(result_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 10))
        
        # 创建Treeview显示结果
        tree_frame = ttk.Frame(result_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        self.result_tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set,
            show='tree headings'
        )
        self.result_tree.pack(fill=tk.BOTH, expand=True)
        
        y_scrollbar.config(command=self.result_tree.yview)
        x_scrollbar.config(command=self.result_tree.xview)
        
        # ==================== 底部状态栏 ====================
        bottom_bar = ttk.Frame(self.root, padding="5")
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Label(
            bottom_bar,
            text="Display: Shortened URIs | Export: Configurable via checkbox above",
            font=("Arial", 8),
            foreground="gray"
        ).pack(side=tk.LEFT)
    
    def generate_default_filename(self):
        """生成默认文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"query_result_{timestamp}"
    
    def execute_query(self):
        """执行SPARQL查询"""
        if not self.ontology_loaded:
            messagebox.showerror("Error", "Ontology not loaded!")
            return
        
        # 获取查询语句
        query_text = self.query_text.get("1.0", tk.END).strip()
        
        if not query_text:
            messagebox.showwarning("Warning", "Please enter a SPARQL query!")
            return
        
        # 更新状态
        self.status_label.config(text="⏳ Executing...", foreground="orange")
        self.root.update()
        
        try:
            # 执行查询
            logger.info("Executing SPARQL query...")
            results = self.graph.query(query_text)
            
            # 转换为DataFrame（保留完整URI和缩写两个版本）
            full_rows = []
            shortened_rows = []
            
            for row in results:
                # 完整URI版本
                full_row = {}
                # 缩写版本
                shortened_row = {}
                
                for var in results.vars:
                    value = row[var]
                    var_name = str(var)
                    
                    if value is not None:
                        # 完整URI
                        full_row[var_name] = str(value)
                        # 缩写URI
                        shortened_row[var_name] = self.shorten_uri(value)
                    else:
                        full_row[var_name] = None
                        shortened_row[var_name] = None
                
                full_rows.append(full_row)
                shortened_rows.append(shortened_row)
            
            # 保存两个版本
            self.current_result_full = pd.DataFrame(full_rows)
            self.current_result_shortened = pd.DataFrame(shortened_rows)
            
            # 显示结果（使用缩写版本）
            self.display_results(self.current_result_shortened)
            
            # 更新状态
            self.status_label.config(
                text=f"✓ Success: {len(self.current_result_full)} rows",
                foreground="green"
            )
            
            logger.info(f"✓ Query returned {len(self.current_result_full)} rows")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query execution failed: {error_msg}")
            
            self.status_label.config(
                text="✗ Query failed",
                foreground="red"
            )
            
            # 显示错误对话框
            messagebox.showerror(
                "Query Error",
                f"Failed to execute query:\n\n{error_msg[:500]}"
            )
    
    def display_results(self, df: pd.DataFrame):
        """显示查询结果"""
        # 清空现有结果
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        if df.empty:
            self.result_info_label.config(text="⚠ Query returned 0 rows")
            return
        
        # 设置列
        columns = list(df.columns)
        self.result_tree['columns'] = columns
        
        # 配置列
        self.result_tree.column("#0", width=50, minwidth=50, stretch=tk.NO)
        self.result_tree.heading("#0", text="Row")
        
        for col in columns:
            self.result_tree.column(col, width=200, minwidth=100, stretch=tk.YES)
            self.result_tree.heading(col, text=col, anchor=tk.W)
        
        # 插入数据（只显示前1000行以保持性能）
        max_display_rows = 1000
        display_df = df.head(max_display_rows)
        
        for idx, row in display_df.iterrows():
            values = [row[col] if pd.notna(row[col]) else "" for col in columns]
            self.result_tree.insert("", tk.END, text=str(idx + 1), values=values)
        
        # 更新信息标签
        info_text = f"📊 {len(df)} rows × {len(columns)} columns"
        if len(df) > max_display_rows:
            info_text += f"  (showing first {max_display_rows} rows)"
        
        self.result_info_label.config(text=info_text)
    
    def export_results(self, format: str):
        """导出查询结果"""
        if self.current_result_full is None or self.current_result_full.empty:
            messagebox.showwarning("Warning", "No results to export!")
            return
        
        # 根据用户选择决定使用哪个版本
        if self.export_shortened_var.get():
            df_to_export = self.current_result_shortened
            uri_format = "shortened"
        else:
            df_to_export = self.current_result_full
            uri_format = "full"
        
        # 获取文件名
        filename = self.filename_entry.get().strip()
        if not filename:
            filename = self.generate_default_filename()
        
        # 移除可能已存在的扩展名
        filename = Path(filename).stem
        
        # 添加扩展名
        if format == 'csv':
            filename = f"{filename}.csv"
        elif format == 'excel':
            filename = f"{filename}.xlsx"
        elif format == 'json':
            filename = f"{filename}.json"
        
        output_path = self.output_dir / filename
        
        try:
            # 导出
            if format == 'csv':
                df_to_export.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif format == 'excel':
                df_to_export.to_excel(output_path, index=False)
            elif format == 'json':
                df_to_export.to_json(output_path, orient='records', indent=2)
            
            logger.info(f"✓ Results exported to: {output_path}")
            
            # 显示成功消息
            messagebox.showinfo(
                "Export Successful",
                f"✓ Exported {len(df_to_export)} rows\n\n"
                f"File: {output_path.name}\n"
                f"Location: {output_path.parent}\n"
                f"URI format: {uri_format}"
            )
            
            # 更新文件名为下一次导出准备
            self.filename_entry.delete(0, tk.END)
            self.filename_entry.insert(0, self.generate_default_filename())
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            messagebox.showerror("Export Error", f"Failed to export results:\n\n{e}")
    
    def clear_query(self):
        """清空查询"""
        self.query_text.delete("1.0", tk.END)
    
    def load_query_from_file(self):
        """从文件加载查询"""
        file_path = filedialog.askopenfilename(
            title="Select SPARQL Query File",
            filetypes=[
                ("SPARQL files", "*.sparql *.rq"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    query_content = f.read()
                
                self.query_text.delete("1.0", tk.END)
                self.query_text.insert("1.0", query_content)
                
                logger.info(f"Loaded query from: {file_path}")
                self.status_label.config(text="✓ Query loaded", foreground="green")
                
            except Exception as e:
                logger.error(f"Failed to load query file: {e}")
                messagebox.showerror("Error", f"Failed to load query:\n\n{e}")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()


# =====================================================
# === 主执行函数
# =====================================================

def main():
    """主执行函数"""
    
    # 配置路径
    TTL_PATH = 'data/outputs/one_shot_extraction/merged_ontology/merged_ontology_OPTIBAR.ttl'
    OUTPUT_PATH = 'data/evaluation_results'
    
    # 检查TTL文件是否存在
    if not Path(TTL_PATH).exists():
        print(f"Error: TTL file not found: {TTL_PATH}")
        print("Please check the file path and try again.")
        return
    
    # 创建并运行GUI
    app = SPARQLQueryGUI(TTL_PATH, OUTPUT_PATH)
    app.run()


if __name__ == "__main__":
    main()