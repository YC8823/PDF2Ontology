import os
from huggingface_hub import snapshot_download

def download_model():
    # 定义保存路径，建议放在 workspace 下以持久化 (如果使用了 Network Volume)
    model_path = "/workspace/models/Dolphin-1.5"
    
    print(f"Downloading ByteDance/Dolphin-1.5 to {model_path}...")
    
    try:
        snapshot_download(
            repo_id="ByteDance/Dolphin-1.5",
            local_dir=model_path,
            local_dir_use_symlinks=False,  # 避免符号链接问题
            resume_download=True
        )
        print("Download completed successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise e

if __name__ == "__main__":
    download_model()