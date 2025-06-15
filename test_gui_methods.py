#!/usr/bin/env python3
"""
Test GUI startup and method availability
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_gui_methods():
    """Test if GUI methods are available"""
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.modern_voice_app import ModernVoiceProcessingApp
        
        # Create app but don't show
        app = QApplication([])
        
        print("🧪 创建GUI实例...")
        window = ModernVoiceProcessingApp()
        
        print("🔍 检查方法是否存在...")
        
        # Check if get_current_card method exists
        if hasattr(window, 'get_current_card'):
            print("  ✅ get_current_card 方法存在")
        else:
            print("  ❌ get_current_card 方法不存在")
            return False
            
        # Check if processors exist
        if hasattr(window, 'processors'):
            print("  ✅ processors 属性存在")
            print(f"     - 处理器数量: {len(window.processors)}")
            for name, processor in window.processors.items():
                status = "已初始化" if processor else "未初始化"
                print(f"     - {name}: {status}")
        else:
            print("  ❌ processors 属性不存在")
            
        # Check other important methods
        methods_to_check = [
            'process_preprocessing',
            'process_source_separation', 
            'process_speaker_separation',
            'process_voice_matching'
        ]
        
        for method_name in methods_to_check:
            if hasattr(window, method_name):
                print(f"  ✅ {method_name} 方法存在")
            else:
                print(f"  ❌ {method_name} 方法不存在")
                
        print("🎉 GUI基本结构测试完成")
        return True
        
    except Exception as e:
        print(f"❌ GUI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔧 测试GUI方法可用性...")
    print("=" * 50)
    
    if test_gui_methods():
        print("\n✅ 测试通过，GUI应该可以正常运行")
        print("🚀 尝试运行: python gui/modern_voice_app.py")
    else:
        print("\n❌ 测试失败，请检查代码")
        
    return True

if __name__ == "__main__":
    main()
