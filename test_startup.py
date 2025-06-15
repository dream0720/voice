#!/usr/bin/env python3
"""
Simple startup test for modern GUI
"""

import sys
import os
from pathlib import Path

def test_basic_startup():
    """Test basic startup without showing GUI"""
    try:
        print("🧪 测试基本启动...")
        
        # Add project root to path
        sys.path.append(str(Path(__file__).parent))
        
        # Test imports
        print("  📦 测试导入...")
        from PyQt6.QtWidgets import QApplication
        from gui.modern_voice_app import ModernVoiceProcessingApp
        print("  ✅ 导入成功")
        
        # Create minimal app
        print("  🏗️ 创建应用...")
        app = QApplication([])
        
        # Create window but don't show
        print("  🖼️ 创建窗口...")
        window = ModernVoiceProcessingApp()
        
        print("  🔍 检查窗口属性...")
        print(f"    - 标题: {window.windowTitle()}")
        print(f"    - 大小: {window.size().width()}x{window.size().height()}")
        
        # Check if methods exist
        methods_to_check = [
            'get_current_card',
            'process_preprocessing', 
            'process_source_separation',
            'process_speaker_separation',
            'process_voice_matching'
        ]
        
        missing_methods = []
        for method in methods_to_check:
            if hasattr(window, method):
                print(f"    ✅ {method}")
            else:
                print(f"    ❌ {method}")
                missing_methods.append(method)
        
        if missing_methods:
            print(f"  ⚠️ 缺少方法: {missing_methods}")
        else:
            print("  ✅ 所有方法都存在")
            
        print("✅ 基本启动测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 启动测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_show_window():
    """Test showing the window"""
    try:
        print("\n🖼️ 测试显示窗口...")
        
        sys.path.append(str(Path(__file__).parent))
        from PyQt6.QtWidgets import QApplication
        from gui.modern_voice_app import ModernVoiceProcessingApp
        
        app = QApplication([])
        window = ModernVoiceProcessingApp()
        
        # Try to show window briefly
        window.show()
        print("✅ 窗口显示成功")
        
        # Process events briefly
        app.processEvents()
        
        # Close immediately
        window.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 窗口显示测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Modern GUI 启动诊断")
    print("=" * 50)
    
    # Test basic startup
    if not test_basic_startup():
        print("\n❌ 基本启动失败")
        return False
    
    # Test window display
    if not test_show_window():
        print("\n❌ 窗口显示失败")
        return False
    
    print("\n🎉 所有测试通过！GUI应该可以正常运行")
    print("💡 尝试运行: python gui/modern_voice_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    input("\n按Enter键退出...")  # Keep window open
    sys.exit(0 if success else 1)
