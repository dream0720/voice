#!/usr/bin/env python3
"""
Quick validation test for the modern GUI
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def validate_gui_class():
    """Validate the GUI class structure"""
    try:
        print("🔍 验证GUI类结构...")
        
        from PyQt6.QtWidgets import QApplication
        from gui.modern_voice_app import ModernVoiceProcessingApp
        
        # Create minimal app for testing
        app = QApplication([])
        
        print("   🏗️ 创建GUI实例...")
        window = ModernVoiceProcessingApp()
        
        # Check critical methods
        critical_methods = [
            'get_current_card',
            'process_preprocessing',
            'view_preprocessing_results',
            'open_output_folder'
        ]
        
        print("   🧪 检查关键方法...")
        for method_name in critical_methods:
            if hasattr(window, method_name):
                method = getattr(window, method_name)
                if callable(method):
                    print(f"     ✅ {method_name}: 可调用")
                else:
                    print(f"     ❌ {method_name}: 不可调用")
                    return False
            else:
                print(f"     ❌ {method_name}: 不存在")
                return False
        
        # Check processors
        print("   🔧 检查处理器...")
        if hasattr(window, 'processors') and isinstance(window.processors, dict):
            print(f"     ✅ processors字典存在，包含{len(window.processors)}个项目")
            for name, processor in window.processors.items():
                status = "已初始化" if processor else "未初始化"
                print(f"       - {name}: {status}")
        else:
            print("     ❌ processors属性无效")
            return False
            
        print("   ✅ GUI类结构验证通过")
        return True
        
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("🚀 快速验证现代化GUI修复...")
    print("=" * 50)
    
    if validate_gui_class():
        print("\n🎉 验证成功！GUI应该可以正常运行了")
        print("💡 现在你可以：")
        print("   1. 运行 'python gui/modern_voice_app.py' 启动GUI")
        print("   2. 或者运行 'run_modern_gui_debug.bat' 进行调试启动")
        print("   3. 选择音频文件并测试预处理功能")
        return True
    else:
        print("\n❌ 验证失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
