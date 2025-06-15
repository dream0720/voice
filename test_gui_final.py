#!/usr/bin/env python3
"""
最终GUI测试脚本 - 测试模块切换和可视化功能
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import QTimer
    from gui.modern_voice_app import ModernVoiceProcessingApp
    print("✅ GUI导入成功")
except ImportError as e:
    print(f"❌ GUI导入失败: {e}")
    sys.exit(1)


class TestGUIFinal(unittest.TestCase):
    """GUI最终测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
    
    def test_app_creation(self):
        """测试应用创建"""
        try:
            app = ModernVoiceProcessingApp()
            self.assertIsNotNone(app)
            print("✅ 应用创建成功")
        except Exception as e:
            self.fail(f"❌ 应用创建失败: {e}")
    
    def test_module_switching(self):
        """测试模块切换"""
        try:
            app = ModernVoiceProcessingApp()
            
            # 测试切换到每个模块
            modules = [0, 1, 2, 3]  # 预处理、音源分离、人声分离、人声匹配
            module_names = ["音频预处理", "音源分离", "说话人分离", "人声匹配"]
            
            for i, module_id in enumerate(modules):
                app.show_module(module_id)
                self.assertEqual(app.current_module, module_id)
                print(f"✅ 模块 {i}: {module_names[i]} 切换成功")
                
                # 验证模块容器有内容
                self.assertIsNotNone(app.module_container.widget())
                
        except Exception as e:
            self.fail(f"❌ 模块切换测试失败: {e}")
    
    def test_visualization_panel(self):
        """测试可视化面板"""
        try:
            app = ModernVoiceProcessingApp()
            
            # 检查可视化面板是否存在
            self.assertTrue(hasattr(app, 'viz_panel'))
            self.assertIsNotNone(app.viz_panel)
            print("✅ 可视化面板存在")
            
            # 检查可视化面板布局
            self.assertTrue(hasattr(app.viz_panel, 'content_layout'))
            self.assertIsNotNone(app.viz_panel.content_layout)
            print("✅ 可视化面板布局存在")
            
            # 测试更新可视化（无文件）
            app.update_visualization([])
            print("✅ 空可视化更新测试通过")
            
            # 测试更新可视化（不存在的文件）
            app.update_visualization(['nonexistent.png'])
            print("✅ 不存在文件可视化更新测试通过")
            
        except Exception as e:
            self.fail(f"❌ 可视化面板测试失败: {e}")
    
    def test_console_functionality(self):
        """测试控制台功能"""
        try:
            app = ModernVoiceProcessingApp()
            
            # 检查控制台是否存在
            self.assertTrue(hasattr(app, 'console'))
            self.assertIsNotNone(app.console)
            print("✅ 控制台存在")
            
            # 测试控制台消息
            app.console.append_message("测试消息", "info")
            app.console.append_message("成功消息", "success")
            app.console.append_message("警告消息", "warning")
            app.console.append_message("错误消息", "error")
            print("✅ 控制台消息测试通过")
            
        except Exception as e:
            self.fail(f"❌ 控制台功能测试失败: {e}")
    
    def test_card_creation(self):
        """测试卡片创建"""
        try:
            app = ModernVoiceProcessingApp()
            
            # 测试每个卡片的创建
            cards = [
                app.create_preprocessing_card(),
                app.create_source_separation_card(),
                app.create_speaker_separation_card(),
                app.create_voice_matching_card()
            ]
            
            card_names = ["预处理卡片", "音源分离卡片", "说话人分离卡片", "人声匹配卡片"]
            
            for i, card in enumerate(cards):
                self.assertIsNotNone(card)
                print(f"✅ {card_names[i]}创建成功")
                
                # 检查卡片的基本组件
                self.assertTrue(hasattr(card, 'process_btn'))
                self.assertTrue(hasattr(card, 'view_results_btn'))
                self.assertTrue(hasattr(card, 'open_folder_btn'))
                print(f"✅ {card_names[i]}组件完整")
                
        except Exception as e:
            self.fail(f"❌ 卡片创建测试失败: {e}")


def run_gui_test():
    """运行GUI测试"""
    print("🚀 开始GUI最终测试...")
    print("=" * 50)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGUIFinal)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 50)
    if result.wasSuccessful():
        print("🎉 所有测试通过！GUI已准备就绪")
        return True
    else:
        print("❌ 部分测试失败，请检查问题")
        return False


def interactive_test():
    """交互式测试"""
    print("\n🎯 启动交互式GUI测试...")
    
    try:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # 创建GUI应用
        gui_app = ModernVoiceProcessingApp()
        gui_app.show()
        
        print("✅ GUI应用已启动")
        print("📝 请手动测试以下功能：")
        print("1. 侧边栏导航切换")
        print("2. 各模块卡片显示")
        print("3. 控制台输出")
        print("4. 可视化面板")
        print("5. 可折叠面板功能")
        
        # 设置定时器自动关闭（10秒后）
        timer = QTimer()
        timer.timeout.connect(app.quit)
        timer.start(10000)  # 10 seconds
        
        # 运行应用
        app.exec()
        print("✅ 交互式测试完成")
        
    except Exception as e:
        print(f"❌ 交互式测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行自动化测试
    test_success = run_gui_test()
    
    if test_success:
        # 如果自动化测试通过，运行交互式测试
        interactive_test()
    else:
        print("❌ 自动化测试失败，跳过交互式测试")
        sys.exit(1)
