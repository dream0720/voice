"""
系统环境测试脚本
Test System Environment

检查所有必需的依赖项是否正确安装
"""
import sys
import importlib
from pathlib import Path

def test_imports():
    """测试关键模块导入"""
    required_modules = [
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('librosa', 'Librosa'),
        ('soundfile', 'SoundFile'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('PyQt6', 'PyQt6'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas'),
        ('demucs', 'Demucs'),
        ('pyannote.audio', 'Pyannote Audio'),
    ]
    
    print("=" * 60)
    print("Voice Processing System - Environment Test")
    print("=" * 60)
    print()
    
    success_count = 0
    total_count = len(required_modules)
    
    for module_name, display_name in required_modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {display_name:.<30} {version}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {display_name:.<30} NOT FOUND ({e})")
    
    print()
    print(f"Status: {success_count}/{total_count} modules available")
    
    if success_count == total_count:
        print("🎉 All dependencies are correctly installed!")
        return True
    else:
        print("⚠️  Some dependencies are missing. Please install them using:")
        print("   Windows: install_pytorch.bat")
        print("   Linux/Mac: bash install_pytorch.sh")
        return False

def test_torch_cuda():
    """测试PyTorch CUDA支持"""
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch not installed")

def test_project_structure():
    """测试项目结构"""
    print("\n" + "=" * 60)
    print("Project Structure Test")
    print("=" * 60)
    
    required_dirs = [
        'modules/preprocessing',
        'modules/source_separation', 
        'modules/speaker_separation',
        'modules/voice_matching',
        'modules/utils',
        'gui',
        'input',
        'reference',
        'output/preprocessing',
        'output/demucs_output',
        'output/speaker_output',
        'output/final_output'
    ]
    
    required_files = [
        'main.py',
        'voice_processing_pipeline.py',
        'requirements.txt',
        'README.md',
        'modules/preprocessing/audio_preprocessor.py',
        'modules/source_separation/demucs_separator.py',
        'modules/speaker_separation/speaker_separator.py',
        'modules/voice_matching/voice_matcher.py',
        'modules/utils/audio_utils.py',
        'gui/voice_processing_app.py'
    ]
    
    print("Checking directories...")
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (missing)")
    
    print("\nChecking files...")
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")

def test_gui():
    """测试GUI是否可以启动"""
    print("\n" + "=" * 60)
    print("GUI Test")
    print("=" * 60)
    
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # 尝试导入GUI模块
        from gui.voice_processing_app import VoiceProcessingApp
        print("✓ GUI components imported successfully")
        
        # 不实际启动GUI，只测试导入
        print("✓ GUI can be initialized (test mode)")
        return True
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("Starting system environment test...\n")
    
    # 测试模块导入
    imports_ok = test_imports()
    
    # 测试CUDA支持
    test_torch_cuda()
    
    # 测试项目结构
    test_project_structure()
    
    # 测试GUI
    gui_ok = test_gui()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if imports_ok and gui_ok:
        print("🎉 System is ready! You can run the application with:")
        print("   python main.py")
        return 0
    else:
        print("⚠️  System needs attention. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
