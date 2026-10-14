@echo off
echo ========================================================
echo   Starting Deepfake Detection Framework - Backend API
echo ========================================================
cd /d "%~dp0backend"

echo Checking Python environment...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo Starting Uvicorn server on http://127.0.0.1:8000 ...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
