@echo off
:: 设置控制台编码为 UTF-8，防止中文乱码
chcp 65001 >nul
setlocal

:: 获取当前脚本所在目录
set "CURRENT_DIR=%~dp0"

:: 自动探测本地 python_env 路径
set "PYTHON_EXE=%CURRENT_DIR%python_env\python.exe"
if exist "%CURRENT_DIR%python_env\Scripts\python.exe" (
    set "PYTHON_EXE=%CURRENT_DIR%python_env\Scripts\python.exe"
)

:: 检查是否真的找到了 python
if not exist "%PYTHON_EXE%" (
    echo [错误] 找不到 Python 解释器！
    echo 请先运行本目录下的“一键初始化环境.bat”来配置运行环境。
    echo.
    pause
    exit /b 1
)

:: 设置 Streamlit 运行命令
echo =========================================
echo   正在启动河源不动产合同识别系统...
echo   请勿关闭此窗口，浏览器将自动打开...
echo =========================================

:: 运行 Streamlit
:: --global.developmentMode=false 隐藏开发者选项
"%PYTHON_EXE%" -m streamlit run "%CURRENT_DIR%pdfapp_web.py" --global.developmentMode=false --server.headless=true --browser.gatherUsageStats=false

endlocal
pause