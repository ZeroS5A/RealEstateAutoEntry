@echo off
:: 设置控制台编码为 UTF-8，防止中文乱码
chcp 65001 >nul
title 自动录入终端 - 环境一键初始化脚本

echo =========================================
echo   正在初始化运行环境，请保持网络畅通...
echo =========================================
echo.

:: 1. 检查 python_env 文件夹是否存在，如果不存在则尝试创建
if not exist ".\python_env\" (
    echo [提示] 未检测到 python_env 文件夹，正在使用系统 Python 创建虚拟环境...
    python -m venv python_env
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败！
        echo 请确认您的电脑已安装 Python 并已添加至系统环境变量 PATH。
        echo.
        pause
        exit /b 1
    )
    echo [成功] 虚拟环境创建完毕！
    echo.
)

:: 2. 自动探测本地 python_env 路径
set PYTHON_EXE=.\python_env\python.exe
if exist ".\python_env\Scripts\python.exe" (
    set PYTHON_EXE=.\python_env\Scripts\python.exe
)

:: 检查是否真的找到了 python
if not exist "%PYTHON_EXE%" (
    echo [错误] 找不到 Python 解释器！
    echo 请检查当前目录下是否存在 python_env 文件夹，或检查环境是否完整。
    echo 期望路径: %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

echo [成功] 检测到本地 Python 环境: %PYTHON_EXE%
echo.

:: 3. 升级 pip (使用国内清华源加速)
echo [1/2] 正在更新基础包管理器 (pip)...
"%PYTHON_EXE%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 4. 安装核心依赖
echo.
echo [2/2] 正在下载并安装项目依赖包 (这一步可能需要几分钟时间)...
:: 注意：这里必须安装 opencv-contrib-python 才能支持微信二维码引擎
"%PYTHON_EXE%" -m pip install streamlit opencv-contrib-python rapidocr_onnxruntime Pillow numpy requests urllib3 PyMuPDF pandas selenium webdriver_manager -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo =========================================
echo   恭喜！所有依赖安装完成。
echo   现在您可以启动运行您的系统了。
echo =========================================
pause