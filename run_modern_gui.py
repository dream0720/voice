#!/usr/bin/env python3
"""
Quick launcher for Modern Voice Processing Suite GUI
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from gui.modern_voice_app import main
    
    if __name__ == "__main__":
        print("🚀 启动现代化 Voice Processing Suite GUI...")
        print("📂 工作目录:", current_dir)
        print("🎨 界面特性:")
        print("  - 侧边栏导航")
        print("  - 现代化扁平设计")
        print("  - 可折叠控制台和可视化面板")
        print("  - 实时进度反馈")
        print("  - 集成所有处理模块")
        print()
        
        main()
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有依赖已安装:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ 启动失败: {e}")
    sys.exit(1)
