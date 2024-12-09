@echo off
chcp 65001 >nul
@setlocal

:: 设置控制台颜色为深色主题
color 0B

:: 设置窗口标题
title FireRedTTS WebUI

:: 设置窗口大小和缓冲区
mode con cols=120 lines=30

:: 显示欢迎信息
echo.
echo    ===============================================
echo                FireRedTTS 语音合成系统
echo    ===============================================
echo.

:: 设置 Python 解释器路径（使用相对路径）
set PYTHON=%~dp0redtts\python.exe

:: 检查 Python 解释器是否存在
if not exist "%PYTHON%" (
    echo    [ERROR] Python解释器未找到: %PYTHON%
    echo    [ERROR] 请确保 redtts 环境在当前目录下
    echo    [ERROR] 预期路径: .\redtts\python.exe
    echo.
    pause
    exit /b 1
)

:: 显示启动信息
echo    [INFO] 系统初始化中...
echo    [INFO] Python路径: %PYTHON%
echo.
echo    ===============================================
echo                   正在启动服务...
echo    ===============================================
echo.

:: 启动 WebUI
"%PYTHON%" webui.py

:: 如果程序异常退出，暂停显示错误信息
if errorlevel 1 (
    echo.
    echo    ===============================================
    echo                     错误信息
    echo    ===============================================
    echo.
    echo    [ERROR] 程序运行出错，请检查错误信息
    echo.
    pause
) 