@echo off
title fog-drive-enhancer
cd /d "%~dp0"

cls
echo ========================================
echo     fog-drive-enhancer
echo     Foggy Drive Video Dehazing Tool
echo ========================================
echo.
echo [1/3] Starting backend API...
start "fog-drive-backend" /MIN cmd /c "%~dp0backend\.venv\Scripts\python %~dp0backend\main.py"

timeout /t 4 /nobreak >nul

echo [2/3] Starting frontend...
start "fog-drive-frontend" /MIN cmd /c "cd /d %~dp0frontend && npx vite --host"

timeout /t 3 /nobreak >nul

echo [3/3] Opening browser...
start http://localhost:5173

echo.
echo ========================================
echo   [OK] All services started
echo   Frontend : http://localhost:5173
echo   Backend  : http://localhost:8000
echo   API Docs : http://localhost:8000/docs
echo.
echo   Close this window = Stop all services
echo ========================================
echo.

:wait
pause >nul
goto :wait
