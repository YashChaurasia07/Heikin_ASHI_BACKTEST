@echo off
echo ========================================
echo Heikin Ashi Backend - Quick Start
echo ========================================
echo.

REM Check if MongoDB is running
echo [1/5] Checking MongoDB...
net start | find "MongoDB" >nul
if %errorlevel% neq 0 (
    echo MongoDB is not running. Starting MongoDB...
    net start MongoDB
    if %errorlevel% neq 0 (
        echo Failed to start MongoDB. Please start it manually.
        pause
        exit /b 1
    )
)
echo MongoDB is running.
echo.

REM Check if virtual environment exists
echo [2/5] Checking Python environment...
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo [4/5] Installing dependencies...
pip install -r requirements.txt
echo.

REM Check .env file
if not exist ".env" (
    echo [!] .env file not found. Copying from .env.example...
    copy .env.example .env
    echo Please edit .env file with your configuration.
    pause
)
echo.

REM Start server
echo [5/5] Starting server...
echo.
python main.py

pause
