"""
Start OBS Studio's Virtual Camera via obs-websocket.

OBS 32.x bundles obs-websocket but it must be ENABLED:
  OBS -> Tools -> obs-websocket Settings -> enable "Enable Heartbeat" / server,
  set a port (default 4455) and an optional password.
If obs-websocket isn't enabled, this script prints clear instructions and exits 1
so the .bat can tell the user to click "Start Virtual Camera" manually.
"""
from __future__ import annotations

import sys
import time

HOST, PORT = "localhost", 4455
PASSWORD = ""  # set this if you configured a password in obs-websocket


def main():
    try:
        from obswebsocket import obsws, requests as obsreq
    except Exception as e:  # noqa: BLE001
        print(f"obs-websocket-py not installed: {e}")
        return 2
    for attempt in range(10):
        try:
            ws = obsws(HOST, PORT, PASSWORD)
            ws.connect()
            ws.call(obsreq.StartVirtualCamera())
            ws.disconnect()
            print("OBS Virtual Camera started.")
            return 0
        except Exception as e:  # noqa: BLE001
            if attempt == 0:
                print(f"obs-websocket not reachable yet ({e}); retrying...")
            time.sleep(2)
    print("\nCould not auto-start the OBS Virtual Camera.")
    print("Enable obs-websocket in OBS (Tools > obs-websocket Settings) and/or")
    print("click 'Start Virtual Camera' manually, then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
