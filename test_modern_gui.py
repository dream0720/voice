#!/usr/bin/env python3
"""
Test runner for the modern voice processing GUI
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Test imports
def test_imports():
    """Test if all required modules can be imported"""
    try:
        from gui.modern_voice_app import ModernVoiceProcessingApp
        print("✅ GUI模块导入成功")
        
        from modules.preprocessing.audio_preprocessor import AudioPreprocessor
        print("✅ 音频预处理模块导入成功")
        
        from modules.source_separation.demucs_separator import DemucsSourceSeparator
        print("✅ 音源分离模块导入成功")
        
        from modules.speaker_separation.speaker_separator import SpeakerSeparator
        print("✅ 说话人分离模块导入成功")
        
        from modules.voice_matching.voice_matcher import VoiceMatcher
        print("✅ 人声匹配模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_gui_launch():
    """Test GUI launch"""
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.modern_voice_app import ModernVoiceProcessingApp
        
        app = QApplication(sys.argv)
        window = ModernVoiceProcessingApp()
        
        # Test basic window properties
        print(f"✅ GUI窗口创建成功")
        print(f"   - 窗口标题: {window.windowTitle()}")
        print(f"   - 窗口大小: {window.size().width()}x{window.size().height()}")
        
        # Test module cards
        if hasattr(window, 'cards') and window.cards:
            print(f"   - 功能模块数量: {len(window.cards)}")
            for name, card in window.cards.items():
                print(f"     * {name}: {card.title}")
        
        # Don't actually show the window in test mode
        return True
        
    except Exception as e:
        print(f"❌ GUI启动测试失败: {e}")
        return False

def main():
    """Main test function"""
    print("🔥 开始测试现代化Voice Processing Suite GUI...")
    print("=" * 60)
    
    # Test imports
    print("\n📦 测试模块导入...")
    if not test_imports():
        print("❌ 导入测试失败，请检查依赖")
        return False
    
    # Test GUI
    print("\n🖼️ 测试GUI启动...")
    if not test_gui_launch():
        print("❌ GUI测试失败")
        return False
        
    print("\n✅ 所有测试通过！")
    print("🚀 可以运行 'python gui/modern_voice_app.py' 启动应用")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
