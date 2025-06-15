"""
快速测试GUI修复效果
"""
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def quick_test():
    """快速测试GUI"""
    try:
        print("🚀 启动GUI测试...")
        
        from gui.modern_voice_app import ModernVoiceProcessingApp
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        
        # 创建主窗口
        window = ModernVoiceProcessingApp()
        window.show()
        
        print("✅ GUI启动成功!")
        print("🔧 请测试模块切换功能，然后关闭窗口")
        
        # 运行应用
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
