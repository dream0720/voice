"""
测试GUI模块切换和卡片管理
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_module_switching():
    """测试模块切换功能"""
    try:
        print("🧪 测试GUI模块切换...")
        
        from gui.modern_voice_app import ModernVoiceProcessingApp
        from PyQt6.QtWidgets import QApplication
        
        # 创建应用实例
        app = QApplication([])
        main_window = ModernVoiceProcessingApp()
        
        print("✅ GUI应用创建成功")
        
        # 测试模块切换
        modules = [
            (0, "音频预处理", "preprocessing_card"),
            (1, "音源分离", "source_separation_card"),
            (2, "说话人分离", "speaker_separation_card"),
            (3, "人声匹配", "voice_matching_card")
        ]
        
        for module_id, module_name, card_attr in modules:
            print(f"\n📋 测试模块 {module_id}: {module_name}")
            
            # 显示模块
            main_window.show_module(module_id)
            
            # 检查卡片属性是否存在
            if hasattr(main_window, card_attr):
                card = getattr(main_window, card_attr)
                if card and not card.isWidgetType() or card.parent():
                    print(f"  ✅ {card_attr} 创建成功并有效")
                else:
                    print(f"  ⚠️  {card_attr} 存在但可能无效")
            else:
                print(f"  ❌ {card_attr} 属性缺失")
            
            # 检查当前模块设置
            if main_window.current_module == module_id:
                print(f"  ✅ current_module 正确设置为 {module_id}")
            else:
                print(f"  ❌ current_module 错误: 期望 {module_id}, 实际 {main_window.current_module}")
        
        # 测试重复切换
        print(f"\n🔄 测试重复切换...")
        for i in range(3):
            print(f"  轮次 {i+1}:")
            for module_id, module_name, card_attr in modules[:2]:  # 只测试前两个模块
                main_window.show_module(module_id)
                card = getattr(main_window, card_attr, None)
                if card:
                    print(f"    ✅ {module_name} 卡片仍然有效")
                else:
                    print(f"    ❌ {module_name} 卡片已失效")
        
        # 测试get_current_card方法
        print(f"\n🎯 测试get_current_card方法...")
        for module_id, module_name, _ in modules:
            main_window.current_module = module_id
            main_window.show_module(module_id)
            card = main_window.get_current_card()
            if card:
                print(f"  ✅ 模块 {module_id} ({module_name}): get_current_card 返回有效卡片")
            else:
                print(f"  ❌ 模块 {module_id} ({module_name}): get_current_card 返回None")
        
        print(f"\n🎉 模块切换测试完成!")
        
        # 关闭应用
        app.quit()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_module_switching()
    if success:
        print("\n✅ 测试通过!")
    else:
        print("\n❌ 测试失败!")
    
    input("\n按回车键退出...")
