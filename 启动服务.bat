@echo off
title ChenMaster-Backend
cd /d "%~dp0"

echo ================================================
echo   陈大师后端服务
echo   保持这个窗口开着 = 服务在运行
echo   关闭这个窗口   = 停止服务
echo ================================================
echo.

rem ---- 先关掉还占着 8000 端口的旧服务，避免"端口被占用"这个坑 ----
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [清理] 关掉旧的 python 服务 PID %%a
    taskkill /F /PID %%a >nul 2>&1
)
rem ---- 等一下，确保端口真的被释放（ping 当计时器用，比 timeout 更稳，不依赖输入重定向）----
ping -n 3 127.0.0.1 >nul

rem ---- 嵌入模型走本地缓存：启动更快，也不依赖网络 ----
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo [启动] 正在启动，首次要加载模型，约 15-30 秒
echo [启动] 看到 Application startup complete 就是好了
echo [启动] 然后手机浏览器打开 http://192.168.0.108:8000/h5
echo.
"C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe" server.py

echo.
echo [已停止] 服务退出了。按任意键关掉这个窗口。
pause >nul
