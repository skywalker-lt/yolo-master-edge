#!/usr/bin/env python3
"""Render test.icon (Icon Composer) -> a macOS squircle AppIcon + a valid .icns.
Approximates the Icon Composer design: white->pink vertical gradient squircle (per icon.json)
with the YOLO-ML-Master glyph composited on top + a soft shadow."""
import io, struct, numpy as np
from PIL import Image, ImageFilter

# run from the `mac/` directory:  python3 scripts/make_icon.py
SRC   = "AppIcon.icon/Assets/icon829.png"
OUT_PNG = "Resources/AppIcon.png"
OUT_ICNS = "Resources/AppIcon.icns"
N = 1024

# ---- white -> pink gradient (icon.json: extended-gray 1,1  ->  display-p3 0.972,0.584,0.865) ----
top    = np.array([255, 255, 255], np.float32)
bottom = np.array([250, 150, 221], np.float32)   # P3 pink -> approx sRGB
ys = np.clip(np.linspace(0, 1, N) / 0.70, 0, 1)  # gradient reaches full pink at 70% height
grad = (top[None, :] * (1 - ys)[:, None] + bottom[None, :] * ys[:, None])
grad = np.repeat(grad[:, None, :], N, axis=1).astype(np.uint8)   # (N,N,3)
bg = Image.fromarray(grad, "RGB").convert("RGBA")

# ---- squircle (superellipse) mask, centered with margin ----
margin = 74                      # transparent border around the squircle (macOS shelf padding)
S = N - 2 * margin
n_exp = 5.0                      # squircle exponent (Apple-ish continuous corner)
ax = np.linspace(-1, 1, S)
xx, yy = np.meshgrid(ax, ax)
inside = (np.abs(xx) ** n_exp + np.abs(yy) ** n_exp) <= 1.0
# anti-alias the edge
r = (np.abs(xx) ** n_exp + np.abs(yy) ** n_exp)
alpha = np.clip((1.02 - r) / 0.04, 0, 1)          # soft 1-px-ish falloff
sq_alpha = np.zeros((N, N), np.float32)
sq_alpha[margin:margin+S, margin:margin+S] = alpha
mask = Image.fromarray((sq_alpha * 255).astype(np.uint8), "L")

icon = Image.new("RGBA", (N, N), (0, 0, 0, 0))
icon.paste(bg, (0, 0), mask)

# ---- soft drop shadow beneath the squircle (subtle, for the macOS shelf look) ----
shadow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
sh_alpha = (sq_alpha * 90).astype(np.uint8)
shadow.putalpha(Image.fromarray(sh_alpha, "L"))
shadow = shadow.filter(ImageFilter.GaussianBlur(22))
shadow = shadow.transform((N, N), Image.AFFINE, (1, 0, 0, 0, 1, 16))  # nudge down
canvas = Image.new("RGBA", (N, N), (0, 0, 0, 0))
canvas.alpha_composite(shadow)
canvas.alpha_composite(icon)

# ---- glyph layer, scaled to fit inside the squircle, with its own soft shadow ----
glyph = Image.open(SRC).convert("RGBA")
target_w = int(S * 0.74)
scale = target_w / glyph.width
glyph = glyph.resize((target_w, int(glyph.height * scale)), Image.LANCZOS)
gx = (N - glyph.width) // 2
gy = (N - glyph.height) // 2
gshadow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
gshadow.paste(glyph, (gx, gy + 6), glyph)
ga = np.array(gshadow.split()[-1], np.float32) * 0.28
gshadow.putalpha(Image.fromarray(ga.astype(np.uint8), "L"))
gshadow = Image.new("RGBA", (N, N), (0, 0, 0, 0))  # rebuild as pure shadow color
tmp = Image.new("RGBA", (N, N), (0, 0, 0, 0)); tmp.paste(glyph, (gx, gy + 8), glyph)
sa = (np.array(tmp.split()[-1], np.float32) * 0.30).astype(np.uint8)
gshadow.putalpha(Image.fromarray(sa, "L"))
gshadow = gshadow.filter(ImageFilter.GaussianBlur(9))
canvas.alpha_composite(gshadow)
canvas.alpha_composite(Image.new("RGBA", (N, N), (0,0,0,0)))
tmp2 = Image.new("RGBA", (N, N), (0, 0, 0, 0)); tmp2.paste(glyph, (gx, gy), glyph)
canvas.alpha_composite(tmp2)

canvas.save(OUT_PNG)
print("wrote", OUT_PNG, canvas.size)

# ---- pack a valid .icns (PNG-encoded entries) ----
def png_bytes(size):
    b = io.BytesIO(); canvas.resize((size, size), Image.LANCZOS).save(b, "PNG"); return b.getvalue()

types = {'icp4':16,'icp5':32,'icp6':64,'ic07':128,'ic08':256,'ic09':512,'ic10':1024,
         'ic11':32,'ic12':64,'ic13':256,'ic14':512}
blob = b''
for t, sz in types.items():
    p = png_bytes(sz)
    blob += t.encode('ascii') + struct.pack('>I', len(p) + 8) + p
icns = b'icns' + struct.pack('>I', len(blob) + 8) + blob
open(OUT_ICNS, 'wb').write(icns)
print("wrote", OUT_ICNS, len(icns), "bytes,", len(types), "sizes")
