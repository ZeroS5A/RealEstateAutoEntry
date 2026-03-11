@echo off
setlocal

:: 获取当前脚本所在目录
set "CURRENT_DIR=%~dp0"

:: 设置 Python 解释器路径
set "PYTHON_EXE=%CURRENT_DIR%python_env\python.exe"

:: 设置 Streamlit 运行命令
echo 正在启动河源不动产合同识别系统...
echo 请勿关闭此窗口，浏览器将自动打开...

:: 运行 Streamlit
:: --global.developmentMode=false 隐藏开发者选项
"%PYTHON_EXE%" -m streamlit run "%CURRENT_DIR%pdfapp_web.py" --global.developmentMode=false --server.headless=true --browser.gatherUsageStats=false

endlocal
pause
```

---

### 最终交付

1.  **测试**：双击 `启动程序.bat`，看是否能正常弹出浏览器并运行。
2.  **分发**：将 `ContractOCR_Portable` 文件夹整个压缩成 `.zip` 发送给同事。
3.  **使用**：同事解压后，双击 `启动程序.bat` 即可使用。

### 替代方案：局域网共享 (最最简单)

如果你的同事和你处于同一个办公室（同一个局域网 WiFi/网线）：

1.  **不需要打包**，也不需要发文件给他们。
2.  你自己在电脑上运行 `streamlit run web_app.py`。
3.  在终端里，你会看到两个地址：
    ```text
    Local URL: http://localhost:8501
    Network URL: http://192.168.1.X:8501