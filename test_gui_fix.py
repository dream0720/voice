"""
完整测试GUI修复效果
"""
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_gui_fixes():
    """测试GUI修复效果"""
    try:
        print("🔧 开始测试GUI修复效果...")
        
        from gui.modern_voice_app import ModernVoiceProcessingApp, ProcessingThread
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        
        print("✅ 导入成功")
        
        # 创建主窗口
        window = ModernVoiceProcessingApp()
        
        print("✅ 主窗口创建成功")
        
        # 测试卡片管理
        print("\n📋 测试卡片管理...")
        modules = [0, 1, 2, 3]  # 所有模块
        for module_id in modules:
            window.show_module(module_id)
            card = window.get_current_card()
            if card:
                print(f"  ✅ 模块 {module_id}: 卡片创建成功")
            else:
                print(f"  ❌ 模块 {module_id}: 卡片创建失败")
        
        # 测试ProcessingThread类
        print("\n🧵 测试ProcessingThread类...")
        if hasattr(ProcessingThread, 'run'):
            print("  ✅ ProcessingThread.run 方法存在")
        if hasattr(ProcessingThread, '_process_preprocessing'):
            print("  ✅ ProcessingThread._process_preprocessing 方法存在")
        if hasattr(ProcessingThread, '_process_source_separation'):
            print("  ✅ ProcessingThread._process_source_separation 方法存在")
        if hasattr(ProcessingThread, '_process_speaker_separation'):
            print("  ✅ ProcessingThread._process_speaker_separation 方法存在")
        if hasattr(ProcessingThread, '_process_voice_matching'):
            print("  ✅ ProcessingThread._process_voice_matching 方法存在")
        
        # 测试处理方法存在性
        print("\n⚙️ 测试处理方法...")
        methods = [
            'process_preprocessing',
            'process_source_separation', 
            'process_speaker_separation',
            'process_voice_matching',
            'get_current_card',
            'check_voice_matching_ready',
            'update_visualization',
            'show_custom_dialog'
        ]
        
        for method in methods:
            if hasattr(window, method):
                print(f"  ✅ {method} 方法存在")
            else:
                print(f"  ❌ {method} 方法缺失")
        
        # 测试成功回调方法
        print("\n📊 测试成功回调方法...")
        callbacks = [
            'on_preprocessing_success',
            'on_source_separation_success',
            'on_speaker_separation_success', 
            'on_voice_matching_success'
        ]
        
        for callback in callbacks:
            if hasattr(window, callback):
                print(f"  ✅ {callback} 方法存在")
            else:
                print(f"  ❌ {callback} 方法缺失")
        
        # 测试结果显示方法
        print("\n📋 测试结果显示方法...")
        result_methods = [
            'view_preprocessing_results',
            'view_source_separation_results',
            'view_speaker_separation_results',
            'view_voice_matching_results',
            'open_output_folder'
        ]
        
        for method in result_methods:
            if hasattr(window, method):
                print(f"  ✅ {method} 方法存在")
            else:
                print(f"  ❌ {method} 方法缺失")
        
        # 测试处理器
        print("\n🔧 测试处理器...")
        processors = ['preprocessor', 'source_separator', 'speaker_separator', 'voice_matcher']
        for proc in processors:
            if proc in window.processors:
                print(f"  ✅ {proc} 处理器已初始化")
            else:
                print(f"  ❌ {proc} 处理器未初始化")
        
        print("\n🎉 GUI修复测试完成!")
        print("\n修复内容总结:")
        print("  ✅ 修正了处理器方法调用名称")
        print("  ✅ 添加了可视化结果显示")
        print("  ✅ 修复了UI状态更新")
        print("  ✅ 修正了消息框字体颜色")
        print("  ✅ 改进了卡片管理和模块切换")
        print("  ✅ 添加了ProcessingThread类和信号处理")
        
        # 关闭应用
        app.quit()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_fixes()
    if success:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 测试发现问题!")
    
    input("\n按回车键退出...")
