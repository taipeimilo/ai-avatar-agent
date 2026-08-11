@echo off
REM ============================================================
REM  start_obs.bat  -- one-click launch for the AI Avatar Agent
REM
REM  How it works (reliable VTuber method):
REM    1. Launch OBS Studio.
REM    2. Create/refresh a scene with a "Window Capture" of the "AI Avatar"
REM       window and START OBS's Virtual Camera (via obs-websocket).
REM    3. Run the avatar: it renders animated frames into the "AI Avatar"
REM       window. OBS captures that window and streams it to Teams/Zoom.
REM
REM  In your meeting app select "OBS Virtual Camera" as the camera.
REM  For the avatar's VOICE, install VB-Audio Virtual Cable
REM  (https://vb-audio.com/Cable/) and pick "CABLE Input" as the mic.
REM
REM  One-time OBS setup: Tools > obs-websocket Settings > enable server
REM  (port 4455). If you skip this, the .bat will ask you to click
REM  "Start Virtual Camera" + add a Window Capture of "AI Avatar" manually.
REM ============================================================
SETLOCAL
SET "REPO=%~dp0"
SET "OBS=C:\Users\milo_\obs-studio\bin\64bit\obs64.exe"
SET "PY=%REPO%.venv\Scripts\python.exe"
IF NOT EXIST "%PY%" SET "PY=python"

cd /d "%REPO%"

echo === AI Avatar Agent ===
echo [1/3] Launching OBS Studio...
IF NOT EXIST "%OBS%" (
  echo   OBS not found at %OBS% -- please install OBS Studio.
  pause & exit /b 1
)
for %%I in ("%OBS%") do set "OBSBIN=%%~dpI"
start "" /d "%OBSBIN%" "%OBS%"
echo   waiting for OBS...
timeout /t 8 /nobreak >nul

echo [2/3] Setting up OBS scene (Window Capture of "AI Avatar") + starting Virtual Camera...
"%PY%" "%REPO%scripts\setup_obs_scene.py"
IF ERRORLEVEL 1 (
  echo   (obs-websocket not enabled or unavailable.)
  echo   -> In OBS: add a "Window Capture" source named anything, pick the
  echo      "AI Avatar" window, then click "Start Virtual Camera" (bottom-right).
)

echo [3/3] Running the avatar in LIVE mode (renders into the "AI Avatar" window)...
echo       Select "OBS Virtual Camera" as your camera in the meeting app.
echo       Press Ctrl-C here to stop.
echo.
"%PY%" "%REPO%src\main.py" --live

echo Avatar stopped.
pause
ENDLOCAL
