# **Dolphin 1.5 RunPod Remote Deployment Guide**

This document details the complete process for deploying the Dolphin 1.5 multimodal document analysis model on the RunPod GPU cloud platform. This guide is specifically optimized for **CUDA environment compatibility**, addressing common "Torch CPU/GPU version conflict" issues.

## **1\. Hardware & Environment Selection (Critical)**

When creating a Pod, you must select the configuration according to the following standards; otherwise, the environment may fail to start, or CUDA may be unavailable.

### **1.1 GPU Selection**

Dolphin 1.5 is a multimodal large model with specific VRAM requirements.

* **Recommended Configuration**: NVIDIA RTX 3090 / RTX 4090 (24GB VRAM) or higher (A6000, A100).  
* **Minimum Configuration**: 24GB VRAM is the baseline for safe operation. VRAM requirements will increase if processing high-resolution PDFs or using large Batch Sizes.  
* **Container Disk**: Recommended at least **50GB** (for storing environment dependencies).  
* **Volume Disk**: Recommended at least **50GB** (for persistent storage of model weights to avoid loss upon restart).

### **1.2 Image Template**

**This is the most critical step.** To avoid mismatches between CUDA driver versions and PyTorch versions, strictly use the following official template:

* **Template Name**: RunPod PyTorch 2.4.0 (or higher)  
* **Image**: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04  
* **Python Version**: 3.11  
* **CUDA Version**: 12.4.1

**Warning**: Do not use outdated templates (such as CUDA 11.x); otherwise, newer versions of transformers and flash-attn may fail to compile or run.

## **2\. Prepare Deployment Files**

Prepare the following file structure locally for uploading to the /workspace directory on RunPod:

/deployment\_files  
├── setup.sh                 \# Environment initialization script (Core)  
├── runpod\_server.py         \# Remote server code (Stateless version)  
├── download\_model.py        \# Model download script  
├── demo\_page.py             \# Official Dolphin demo logic  
├── demo\_element.py          \# (Optional) Element recognition logic  
├── demo\_layout.py           \# (Optional) Layout analysis logic  
└── utils/                   \# Utility package  
    ├── \_\_init\_\_.py  
    ├── utils.py  
    └── markdown\_utils.py

### **Core Script Descriptions**

#### **setup.sh (Robust Version)**

This script employs a **"Split Installation Strategy"**, forcibly specifying the PyTorch source to prevent pip from automatically pulling the CPU version and overwriting the GPU version.

\# Key snippet preview  
pip install torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/cu124\](https://download.pytorch.org/whl/cu124)  
pip install transformers accelerate ...

#### **runpod\_server.py**

A stateless server that receives uploads via the file field on the /analyze interface. It automatically cleans up temporary files after processing, consuming no server storage.

## **3\. Deployment Steps**

### **Step 1: Connect to Pod**

After the Pod starts, click "Connect" \-\> "Start Web Terminal" or use SSH to connect.

### **Step 2: Upload Files**

We recommend using one of the following two methods to upload files:

1. **JupyterLab (Recommended)**: Click "Connect" \-\> "Jupyter Lab" on the Pod, and drag your local folder directly into the /workspace directory in the left file sidebar.  
2. **SCP / SFTP**: Upload using FileZilla or the scp command.

### **Step 3: Execute Initialization**

Run the following in the RunPod terminal:

cd /workspace  
chmod \+x setup.sh  
./setup.sh

**Script Execution Flow:**

1. Installs system-level dependencies (ffmpeg, libgl1, etc.).  
2. **Critical**: Forcibly installs the GPU version of PyTorch from the CUDA 12.4 source.  
3. Installs remaining dependencies like transformers.  
4. Automatically performs a CUDA availability self-check (will error out immediately if it fails).  
5. Automatically downloads the Dolphin-1.5 model (skips if it already exists).  
6. Starts the server listening on port 8080\.

## **4\. Local Connection (SSH Tunnel)**

To allow local code to access the remote server, you need to establish an SSH tunnel.

**Command Format:**

ssh \-p \[Remote\_Port\] root@\[Remote\_IP\] \-L 8080:localhost:8080

* **\[Remote\_Port\]** & **\[Remote\_IP\]**: Obtain these from the RunPod console.  
* **\-L 8080:localhost:8080**: Maps your local 8080 port to the remote 8080 port.

Verify Connection:  
Visit http://localhost:8080/health in your local browser. If it returns {"status": "ok", ...}, the tunnel is successfully established.

## **5\. Troubleshooting**

### **Q1: After restarting the Pod, I get CUDA driver initialization failed or CUDA available: False error.**

Cause: Restarting a Pod resets the system disk, causing environment loss. If you reinstall dependencies and pip defaults to downloading the CPU version of PyTorch from PyPI, it will overwrite the working GPU version.  
Solution:

1. Do not manually run pip install transformers randomly.  
2. **You Must** run the provided setup.sh. This script contains the \--index-url https://download.pytorch.org/whl/cu124 parameter to forcibly lock the GPU version.

### **Q2: Client reports 422 Unprocessable Entity.**

Cause: The format of the parameters sent by the local request is incorrect. The server expects multipart/form-data format with the field name file.  
Solution: Ensure your client code sends the request using files={'file': open(...)} rather than a JSON body.

### **Q3: No space left on device.**

Cause: The Dolphin model weights are large (\~15GB), and temporary images generated during inference can easily fill up the disk.  
Solution:

1. Use our provided **Stateless Version** of runpod\_server.py. It uses tempfile and automatically deletes intermediate files after processing.  
2. Ensure you allocate sufficient Container Disk (50GB+ recommended) when creating the Pod.

## **6\. Maintenance & Updates**

* **Update Code**: Directly upload new .py files to overwrite those in /workspace. Stop the service with Ctrl+C in the terminal and re-run python3 runpod\_server.py.  
* **Update Dependencies**: If you need to add new packages, edit setup.sh and re-run it to ensure continued compatibility.