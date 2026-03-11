@echo off
echo 正在停止河源不动产合同识别系统...

:: 强制结束所有 python.exe 进程
:: 注意：如果您电脑上有其他 Python 程序在运行，也会被关闭！
taskkill /F /IM python.exe /T

echo.
echo 系统已停止。
ping -n 2 127.0.0.1 > nul