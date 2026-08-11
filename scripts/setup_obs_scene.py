"""
Set up the OBS scene that captures the "AI Avatar" window and start the
OBS Virtual Camera, via obs-websocket.

Run AFTER OBS is launched. Requires obs-websocket enabled in OBS
(Tools > obs-websocket Settings > enable server on port 4455).

What it does:
  - creates/selects a scene "AI Avatar"
  - adds a "window_capture" source bound to the "AI Avatar" window title
  - starts the Virtual Camera
Exits 0 on success, 1 if obs-websocket isn't reachable (caller then tells the
user to do it manually).
"""
from __future__ import annotations

import sys
import time

HOST, PORT = "localhost", 4455
PASSWORD = ""  # set if you configured a websocket password
WINDOW_TITLE = "AI Avatar"
SCENE = "AI Avatar"


def main():
    try:
        from obswebsocket import obsws, requests as obsreq
    except Exception as e:  # noqa: BLE001
        print(f"obs-websocket-py not installed: {e}")
        return 2

    ws = obsws(HOST, PORT, PASSWORD, timeout=5)
    try:
        ws.connect()
    except Exception as e:  # noqa: BLE001
        print(f"obs-websocket not reachable ({e}); enable it in OBS.")
        return 1

    try:
        # Ensure scene exists
        scenes = ws.call(obsreq.GetSceneList()).getScenes()
        names = [s["sceneName"] for s in scenes]
        if SCENE not in names:
            ws.call(obsreq.CreateScene(sceneName=SCENE))
        ws.call(obsreq.SetCurrentProgramScene(sceneName=SCENE))

        # Add/replace window-capture source for the avatar window
        src_name = "AvatarWindow"
        try:
            ws.call(obsreq.RemoveInput(inputName=src_name))
        except Exception:  # noqa: BLE001
            pass
        settings = {
            "window": WINDOW_TITLE,          # match by title substring
            "window_match_priority": 0,       # match title
        }
        ws.call(obsreq.CreateInput(
            sceneName=SCENE,
            inputName=src_name,
            inputKind="window_capture",
            inputSettings=settings,
            sceneItemEnabled=True,
        ))

        # Start the virtual camera
        ws.call(obsreq.StartVirtualCamera())
        print("OBS scene + Virtual Camera started (capturing 'AI Avatar' window).")
        return 0
    finally:
        ws.disconnect()


if __name__ == "__main__":
    sys.exit(main())
