import sys; sys.path.insert(0, r"C:\Users\milo_\ai-avatar-agent")
from src.audio.out import find_device, cable_available, CABLE_NAME
print("cable_available:", cable_available())
idx = find_device(CABLE_NAME)
print("selected CABLE device index:", idx)
import sounddevice as sd
if idx is not None:
    d = sd.query_devices(idx)
    print("selected device name:", d["name"], "ch=", d["max_output_channels"])
from src.face.lightweight import LightweightRenderer
from src.config import load_config
cfg = load_config()
r = LightweightRenderer(cfg, "directml")
idle_open = r.render_idle(blink=False)
idle_closed = r.render_idle(blink=True)
import numpy as np
diff = bool((idle_closed.astype(int) - idle_open.astype(int)).any())
print("blink frame differs from idle:", diff)
