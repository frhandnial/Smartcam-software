@echo off
rem Rebuilds SmartCam's own Python environment (only needed if the venv folder was deleted).
cd /d "%~dp0"
python -m venv venv || (echo Python was not found. Install it from python.org first. & pause & exit /b 1)
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Done. Start SmartCam with run.bat
pause
