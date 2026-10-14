@echo off
echo ========================================================
echo   Deepfake Detection Framework - Model Evaluation Suite
echo   Powered by MLflow and scikit-learn
echo ========================================================
cd /d "%~dp0backend"

echo Checking Python environment...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo Starting evaluation pipeline...
if "%~1"=="" (
    python evaluation/evaluate_models.py --dataset-dir ../dataset/test
) else (
    python evaluation/evaluate_models.py %*
)

echo.
echo Evaluation process completed.
pause
