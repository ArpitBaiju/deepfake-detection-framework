@echo off
echo ===================================================
echo   Starting DAF-Net Live Inspection Studio
echo ===================================================
echo.
echo [1/2] Launching Python FastAPI Backend on port 8000...
start cmd /k "python server.py"

echo [2/2] Launching React Vite Frontend...
cd frontend
start cmd /k "npm run dev"

echo.
echo ===================================================
echo Both Backend and Frontend are starting!
echo Open your browser at: http://localhost:5173
echo ===================================================
