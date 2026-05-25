import pdfplumber
import json

# 替换成你的 Datasheet 路径
pdf_path = "data/test_materials/SS_03.pdf"

def generate_raw_stream(pdf_path, page_num=0):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        
        # 提取所有单词及其坐标信息
        words = page.extract_words(
            x_tolerance=3, 
            y_tolerance=3, 
            keep_blank_chars=False
        )
        
        # 仅保留核心信息，模拟机器视角的"生数据"
        raw_data = []
        for w in words:
            raw_data.append({
                "text": w['text'],
                "bbox": [round(w['x0'], 1), round(w['top'], 1), round(w['x1'], 1), round(w['bottom'], 1)],
                "font": w.get('fontname', 'unknown')
            })
            
        # 打印成 JSON 格式
        print(json.dumps(raw_data, indent=2))

# 运行
generate_raw_stream(pdf_path)