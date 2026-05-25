import os
import shutil
import uvicorn
import torch
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# 导入 demo 模块
from demo_page import DOLPHIN, process_document
# 导入目录创建工具
from utils.utils import setup_output_dirs

# 初始化 FastAPI 应用
app = FastAPI(title="Dolphin 1.5 RunPod Server (Stateless)")

# 全局变量存储模型
model = None
MODEL_PATH = "/workspace/models/Dolphin-1.5"
# 仅保留上传文件的临时目录，处理完即删
TEMP_UPLOAD_DIR = "/workspace/temp_uploads"

# 确保上传目录存在
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def load_model():
    """在服务启动时加载模型到 GPU"""
    global model
    print(f"Loading Dolphin model from {MODEL_PATH}...")
    try:
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, running on CPU might be slow!")
        
        model = DOLPHIN(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Trying to load from HuggingFace hub directly...")
        model = DOLPHIN("ByteDance/Dolphin-1.5")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    无状态分析接口：
    1. 接收文件
    2. 在临时目录处理
    3. 返回 JSON
    4. 自动清理所有服务器端文件
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. 保存上传的文件
    file_location = os.path.join(TEMP_UPLOAD_DIR, file.filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    print(f"Processing request for: {file.filename}")

    try:
        # 2. 使用临时目录作为输出目录 (核心修改)
        # with 语句结束时，temp_dir 及其内容会被自动删除
        with tempfile.TemporaryDirectory() as temp_output_dir:
            
            # 创建必要的子目录结构 (utils.py 需要)
            setup_output_dirs(temp_output_dir)

            # 3. 调用核心处理逻辑
            # 注意：这里我们依然传了 save_dir，让 Dolphin 正常工作（生成图片等），
            # 但这些文件都在 temp_output_dir 里，处理完就丢弃。
            _, raw_results = process_document(
                document_path=file_location,
                model=model,
                save_dir=temp_output_dir,
                max_batch_size=8
            )

            # 4. 构造响应 (仅保留内存中的数据)
            response_data = {
                "pdf_filename": file.filename,
                "total_pages": 0,
                "results_per_page": []
            }

            if file.filename.lower().endswith('.pdf'):
                response_data["total_pages"] = len(raw_results)
                response_data["results_per_page"] = raw_results
            else:
                response_data["total_pages"] = 1
                response_data["results_per_page"] = [{
                    "page_number": 1,
                    "elements": raw_results,
                    "image_filename": file.filename
                }]

            return JSONResponse(content=response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # 5. 清理上传的源文件
        if os.path.exists(file_location):
            try:
                os.remove(file_location)
                print(f"Cleaned up upload: {file_location}")
            except Exception as e:
                print(f"Error cleaning up file {file_location}: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)