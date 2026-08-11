"""
Start OBS Studio's Virtual Camera via obs-websocket.

OBS 32.x bundles obs-websocket but it must be ENABLED:
  OBS -> Tools -> obs-websocket Settings -> enable server on port 4455 (+password).
If obs-websocket isn't enabled, this prints clear instructions and exits 1 so the
.bat can tell the user to click "Start Virtual Camera" manually.

Robustness: every connect/call runs in a thread with a hard timeout so this
NEVER hangs the one-click launcher (a stalled websocket handshake would otherwise
block the user's double-click forever).
"""
from __future__ import annotations

import sys
import time
import threading

HOST, PORT = "localhost", 4455
PASSWORD = ""  # set this if you configured a password in obs-websocket
PER_TRY = 4.0   # seconds before we give up on one connect attempt
MAX_WAIT = 14.0  # total time spent retrying before giving up


def _try_once():
    from obswebsocket import obsws, requests as obsreq
    ws = obsws(HOST, PORT, PASSWORD, timeout=3)
    ws.connect()
    ws.call(obsreq.StartVirtualCamera())
    ws.disconnect()
    return True


def main():
    try:
        import obswebsocket  # noqa: F401  (fail fast if not installed)
    except Exception as e:  # noqa: BLE001
        print(f"obs-websocket-py not installed: {e}")
        return 2

    deadline = time.time() + MAX_WAIT
    first = True
    while time.time() < deadline:
        done = {}
        t = threading.Thread(target=lambda: done.setdefault("r", _try_once()), daemon=True)
        t.start()
        t.join(PER_TRY)
        if "r" in done and done["r"]:
            print("OBS Virtual Camera started.")
            return 0
        if first:
            print("obs-websocket not reachable yet; retrying...")
            first = False
        time.sleep(1)
    print("\nCould not auto-start the OBS Virtual Camera.")
    print("Enable obs-websocket in OBS (Tools > obs-websocket Settings) and/or")
    print("click 'Start Virtual Camera' manually, then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
