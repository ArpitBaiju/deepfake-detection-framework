@echo off
echo =========================================================================
echo   Deepfake Detection Framework - Full Stack Launcher
echo =========================================================================
echo.
echo Launching Backend FastAPI server in a new window...
start "Deepfake Backend API" cmd /k "run_backend.bat"

echo Launching Frontend Vite dev server in a new window...
start "Deepfake Frontend Dashboard" cmd /k "run_frontend.bat"

echo.
echo =========================================================================
echo Both services are starting!
echo - Backend API:    http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)
echo - Frontend App:   http://localhost:5173
echo =========================================================================
echo.
pause
