@echo off
echo ========================================================
echo   Deepfake Detection Framework - MLflow Tracking UI
echo ========================================================
cd /d "%~dp0backend"

echo Checking Python / MLflow...
python -m mlflow --version
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MLflow is not installed in the active environment!
    echo Please run: pip install -r requirements-evaluation.txt
    pause
    exit /b 1
)

set MLFLOW_ALLOW_FILE_STORE=true
set MLFLOW_DISABLE_AGENT_HINT=1

echo.
echo Launching MLflow UI on http://127.0.0.1:5000 ...
echo Tracking store: %~dp0backend\mlruns
echo Press Ctrl+C to terminate the UI server.
echo.
python -m mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
pause
