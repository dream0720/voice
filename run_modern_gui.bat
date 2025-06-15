@echo off
echo 🚀 启动现代化 Voice Processing Suite GUI...
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装或不在 PATH 中
    echo 请安装 Python 3.8+ 并将其添加到系统 PATH
    pause
    exit /b 1
)

REM 检查必要的包
echo 📦 检查依赖包...
python -c "import PyQt6, matplotlib, librosa, torch" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  依赖包可能未完全安装
    echo 正在尝试安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖包安装失败
        pause
        exit /b 1
    )
)

echo ✅ 环境检查完成
echo 🎨 启动现代化界面...
echo.

REM 启动应用
python run_modern_gui.py

pause
