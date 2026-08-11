@echo off
REM ============================================================
REM  start_obs.bat  -- one-click launch for the AI Avatar Agent
REM  Launches OBS Studio, starts its Virtual Camera, then runs
REM  the avatar in LIVE mode (mic -> brain -> tts -> face -> cam).
REM  In Zoom/Teams/Discord, select "OBS Virtual Camera" as the camera.
REM ============================================================
SETLOCAL
SET "REPO=%~dp0"
SET "OBS=C:\Users\milo_\obs-studio\bin\64bit\obs64.exe"
SET "PY=%REPO%.venv\Scripts\python.exe"
IF NOT EXIST "%PY%" SET "PY=python"

cd /d "%REPO%"

echo [1/4] Launching OBS Studio...
IF NOT EXIST "%OBS%" (
  echo   OBS not found at %OBS% -- please install OBS Studio.
  pause
  exit /b 1
)
REM Launch OBS with its OWN bin folder as the working directory so it can
REM resolve data/locale correctly (launching from another CWD breaks it).
for %%I in ("%OBS%") do set "OBSBIN=%%~dpI"
start "" /d "%OBSBIN%" "%OBS%"
echo   waiting for OBS to boot...
timeout /t 8 /nobreak >nul
echo   >> In OBS, click "Start Virtual Camera" (bottom-right) so Teams can see it.
echo   >> (Optional one-time: enable auto-start in Settings > Advanced.)

echo [2/4] Starting OBS Virtual Camera (via obs-websocket)...
"%PY%" "%REPO%scripts\start_obs_camera.py"
IF ERRORLEVEL 1 (
  echo   Could not auto-start the virtual camera.
  echo   -> In OBS, click "Start Virtual Camera" manually, then continue.
  echo   (Tip: enable auto-start in OBS: Settings > Advanced > Automatically start)
  pause
)

echo [3/4] Launching the AI avatar in LIVE mode...
echo   When the avatar window appears, speak -- it will reply with a talking face.
echo   Select "OBS Virtual Camera" in your meeting app.
echo   Press Ctrl-C in this window to stop.
echo [4/4] Running...
"%PY%" "%REPO%src\main.py" --live

echo Avatar stopped.
pause
ENDLOCAL
