#!/usr/bin/env python3
"""Pack Resources/AppIcon.png (a finished 1024x1024 macOS icon, e.g. exported from Icon Composer)
into Resources/AppIcon.icns at every required size. No compositing — the source PNG is used as-is,
so the artwork is never distorted. Run from the `mac/` directory:  python3 scripts/make_icon.py"""
import io, struct
from PIL import Image

SRC  = "Resources/AppIcon.png"
ICNS = "Resources/AppIcon.icns"

master = Image.open(SRC).convert("RGBA")
if master.size != (1024, 1024):
    master = master.resize((1024, 1024), Image.LANCZOS)

# OSType -> pixel size (PNG-encoded entries; macOS picks what it needs)
types = {'icp4':16, 'icp5':32, 'icp6':64, 'ic07':128, 'ic08':256, 'ic09':512, 'ic10':1024,
         'ic11':32, 'ic12':64, 'ic13':256, 'ic14':512}

def png_bytes(size):
    b = io.BytesIO()
    (master if size == 1024 else master.resize((size, size), Image.LANCZOS)).save(b, "PNG")
    return b.getvalue()

blob = b''
for t, sz in types.items():
    p = png_bytes(sz)
    blob += t.encode('ascii') + struct.pack('>I', len(p) + 8) + p
open(ICNS, 'wb').write(b'icns' + struct.pack('>I', len(blob) + 8) + blob)
print(f"wrote {ICNS}: {len(types)} sizes from {SRC}")
