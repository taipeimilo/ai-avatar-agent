@echo off
REM ============================================================
REM  start_obs.bat  -- one-click launch for the AI Avatar Agent
REM
REM  IMPORTANT: the avatar creates the "OBS Virtual Camera" device itself via
REM  pyvirtualcam (it talks to the OBS Virtual Camera DRIVER directly). You do
REM  NOT need to launch OBS Studio or click "Start Virtual Camera" in it -- in
REM  fact doing so CONFLICTS with pyvirtualcam owning the device. Just run this
REM  and pick "OBS Virtual Camera" in your meeting app.
REM
REM  In Zoom/Teams/Discord, select "OBS Virtual Camera" as the camera.
REM  For the avatar's VOICE to be heard, install VB-Audio Virtual Cable
REM  (https://vb-audio.com/Cable/) and select "CABLE Input" as the microphone.
REM ============================================================
SETLOCAL
SET "REPO=%~dp0"
SET "PY=%REPO%.venv\Scripts\python.exe"
IF NOT EXIST "%PY%" SET "PY=python"

cd /d "%REPO%"

echo === AI Avatar Agent ===
echo [1/2] Loading models and starting the talking-head avatar...
echo       (pyvirtualcam will expose "OBS Virtual Camera" automatically)
echo [2/2] Running in LIVE mode. Speak, then pause ~1s; the avatar replies.
echo       Select "OBS Virtual Camera" as your camera in the meeting app.
echo       Press Ctrl-C here to stop.
echo.

"%PY%" "%REPO%src\main.py" --live

echo Avatar stopped.
pause
ENDLOCAL
