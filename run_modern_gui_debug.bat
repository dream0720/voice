@echo off
echo ==============================================
echo 启动现代化Voice Processing Suite GUI
echo ==============================================

cd /d "%~dp0"

echo 当前目录: %CD%
echo.

echo 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: Python未正确安装或不在PATH中
    pause
    exit /b 1
)

echo.
echo 检查必要的Python包...
python -c "import PyQt6; print('PyQt6: OK')" 2>nul || (
    echo 错误: PyQt6未安装
    echo 请运行: pip install PyQt6
    pause
    exit /b 1
)

python -c "import matplotlib; print('matplotlib: OK')" 2>nul || (
    echo 错误: matplotlib未安装
    echo 请运行: pip install matplotlib
    pause
    exit /b 1
)

echo.
echo 启动GUI应用...
echo 如果遇到问题，请查看控制台输出
echo ==============================================
echo.

python gui\modern_voice_app.py

echo.
echo ==============================================
echo 应用已退出
pause
