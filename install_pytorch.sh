#!/bin/bash
echo "Installing PyTorch 2.2.2 with CUDA 12.1 support..."
echo

# 安装指定版本的PyTorch和torchaudio
pip install torch==2.2.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

echo
echo "Installation completed!"
echo "You can now run the application with: python main.py"
