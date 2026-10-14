@echo off
echo ========================================================
echo   Starting Deepfake Detection Framework - Frontend App
echo ========================================================
cd /d "%~dp0frontend"

echo Checking Node.js environment...
node --version
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed or not in PATH!
    pause
    exit /b 1
)

echo Starting Vite Dev Server on http://localhost:5173 ...
npm run dev
pause
