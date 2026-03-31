:: 设置控制台编码为 UTF-8，防止中文乱码
chcp 65001 >nul

@echo off
:: ==========================================
:: 隐藏 CMD 窗口逻辑 (核心修改)
:: ==========================================
if "%1" == "h" goto begin
:: 利用 mshta 调用 vbscript 隐藏运行自身
mshta vbscript:createobject("wscript.shell").run("""%~f0"" h",0)(window.close)&&exit
:begin

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
    :: 静默模式下遇到错误，弹出系统对话框提示，防止用户点击后没反应不知道原因
    mshta vbscript:msgbox("启动失败：找不到Python环境！" ^& vbCrLf ^& "请先运行【一键初始化环境.bat】",16,"环境缺失")(window.close)
    exit /b 1
)

:: 运行 Streamlit
:: --global.developmentMode=false 隐藏开发者选项
"%PYTHON_EXE%" -m streamlit run "%CURRENT_DIR%pdfapp_web.py" --global.developmentMode=false --server.headless=true --browser.gatherUsageStats=false

endlocal