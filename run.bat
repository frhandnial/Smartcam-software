@echo off
rem Starts SmartCam in the background (no console window) and opens the dashboard.
rem Extra options pass straight through, e.g.  run.bat --camera demo
cd /d "%~dp0"
if not exist venv\Scripts\pythonw.exe (
  echo SmartCam's Python environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
start "" venv\Scripts\pythonw.exe smartcam.py %*
