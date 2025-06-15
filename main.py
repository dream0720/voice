"""
现代化人声增强与主人声提取系统
Audio Enhancement & Owner Voice Extraction System

基于信号与系统课程知识点的期末项目
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主程序入口"""
    from gui.voice_processing_app import main as gui_main
    return gui_main()
        

def create_project_structure():
    """创建项目目录结构"""
    dirs_to_create = [
        'input',
        'reference', 
        'output',
        'temp'
    ]
    
    for dir_name in dirs_to_create:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        
        # 创建说明文件
        readme_path = dir_path / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                if dir_name == 'input':
                    f.write("# 输入音频目录\n\n请将需要处理的混合人声音频文件放在此目录中。\n\n支持格式：WAV, MP3, FLAC, M4A, AAC\n")
                elif dir_name == 'reference':
                    f.write("# 参考音频目录\n\n请将参考主人声音频文件放在此目录中。\n\n注意：参考音频应该是目标说话人的清晰语音样本。\n")
                elif dir_name == 'output':
                    f.write("# 输出结果目录\n\n处理后的音频文件将保存在此目录中。\n\n包含：增强后的主人声音频和处理报告。\n")
                else:
                    f.write(f"# {dir_name.title()} 目录\n\n系统临时文件目录，用于存储中间处理结果。\n")

if __name__ == "__main__":
    # 创建项目结构
    create_project_structure()
    
    # 运行主程序
    exit_code = main()
    sys.exit(exit_code or 0)
