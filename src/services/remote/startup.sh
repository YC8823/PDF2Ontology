#!/bin/bash

# 设置遇到错误立即停止
set -e

echo "=== Starting Clean Deployment for Dolphin 1.5 ==="

# 1. 安装系统级依赖 (每次重启都会重置，必须重装)
echo "[1/5] Installing system dependencies (apt)..."
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    wget \
    git \
    python3-pip

# 2. 升级 pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

# 3. 安装 Python 依赖 (拆分安装策略)
# 策略：先从 PyTorch 官方源安装 GPU 版 Torch，再从 PyPI 安装其他库。
# 这样避免了混合源导致的依赖解析混乱，也确保了 transformers 能被找到。

echo "[3.1/5] Installing PyTorch (CUDA 12.4 version)..."
# 使用 --index-url 强制只从 pytorch 源下载，确保绝对是 GPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[3.2/5] Installing other libraries (Transformers, etc.)..."
# 从默认 PyPI 安装其他库。transformers 安装时会检测到 torch 已存在，
# 且版本满足要求，因此不会去重新下载 CPU 版 torch。
pip install \
    transformers \
    accelerate \
    protobuf \
    scipy \
    opencv-python \
    pillow \
    pymupdf \
    uvicorn \
    fastapi \
    python-multipart \
    requests

# 4. 验证环境 (如果这一步失败，脚本会直接报错停止，避免做无用功)
echo "[4/5] Verifying CUDA availability..."
python3 -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); assert torch.cuda.is_available(), 'CRITICAL ERROR: CUDA is NOT available!'"

# 5. 准备目录
echo "[5/5] Setting up workspace directories..."
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/temp_uploads

# 下载模型 (如果不存在)
if [ ! -d "/workspace/models/Dolphin-1.5" ]; then
    echo "Downloading Dolphin-1.5 model..."
    python3 download_model.py
else
    echo "Model found, skipping download."
fi

echo "=== Setup Complete! Starting Server... ==="

# 启动服务器
python3 runpod_server.py