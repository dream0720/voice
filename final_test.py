#!/usr/bin/env python3
"""
Final verification test
"""

import sys
import os
from pathlib import Path

def main():
    """Quick test"""
    try:
        print("🔧 最终验证测试...")
        
        # Add path
        sys.path.append(str(Path(__file__).parent))
        
        # Test compilation
        print("  📝 检查语法...")
        import py_compile
        py_compile.compile('gui/modern_voice_app.py', doraise=True)
        print("  ✅ 语法检查通过")
        
        # Test import
        print("  📦 测试导入...")
        from gui.modern_voice_app import ModernVoiceProcessingApp, main
        print("  ✅ 导入成功")
        
        # Test basic structure
        print("  🏗️ 检查结构...")
        from PyQt6.QtWidgets import QApplication
        app = QApplication([])
        window = ModernVoiceProcessingApp()
        
        # Check methods
        required_methods = [
            'get_current_card',
            'process_preprocessing',
            'process_source_separation', 
            'process_speaker_separation',
            'process_voice_matching',
            'view_preprocessing_results',
            'open_output_folder'
        ]
        
        missing = []
        for method in required_methods:
            if not hasattr(window, method):
                missing.append(method)
        
        if missing:
            print(f"  ❌ 缺少方法: {missing}")
            return False
        else:
            print("  ✅ 所有方法都存在")
            
        print("\n🎉 验证通过！现在可以启动GUI了")
        print("💡 运行: python gui/modern_voice_app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
